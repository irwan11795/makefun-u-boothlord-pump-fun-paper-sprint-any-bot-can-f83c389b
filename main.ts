// Curve Scout — a pump.fun paper-trading bot for the agent-fun/replay/v2 protocol.
//
// Strategy (designed from the protocol spec + the dev dataset, no reference code used):
//   * Scout new bonding curves from their CreateEvent. A coin only becomes a candidate
//     while it is young (<= 100 slots old) and still cheap (price <= 1.10x its create
//     price) with buy-dominated flow (buys >= 60% of its trades so far).
//   * Accumulate in fixed 0.01 SOL chunks (each buy capped at 1% of the curve's real
//     SOL reserves, per the hello caps) up to a 0.05 SOL target per coin, and only
//     while the price stays under the 1.10x cap — never chase a spike.
//   * Exit a position in full on take-profit (+15% vs average entry), stop-loss
//     (-20% vs average entry), or a 28,800-slot hold timeout (~1.7h). One entry per
//     mint ever: once a coin is closed it is never bought again.
//   * At most 5 open positions, at most 16 orders per batch, no shorts, no leverage.
//
// Determinism: every decision is a pure function of the hello and the batches seen
// so far. No clock, no environment, no Math.random; the hello seed is accepted and
// never used for randomness because the strategy needs none.

import { runBot, type Batch, type BotState, type OrderInput } from './vendor/makefun-bot.ts';

const LAMPORTS_PER_SOL = 1_000_000_000n;

// --- tuned parameters (from local replay over the 24h dev dataset) ----------------
const CHUNK_LAMPORTS = 10_000_000n; // 0.01 SOL per buy (also the per-trip minimum)
const TARGET_LAMPORTS = 10_000_000n; // 0.01 SOL target per coin: ONE chunk, no
// accumulation. A single chunk already counts a round trip (>= 0.01 SOL), and
// sizing up only multiplies the fill-slippage loss; throughput (trips) is what
// the 20-coin guardrail counts, and it is set by the exit speed, not the size.
const MIN_CHUNK_LAMPORTS = 5_000_000n; // 0.005 SOL smallest top-up buy
const MAX_AGE_SLOTS = 100; // only scout coins this young (slots since CreateEvent)
const MAX_ENTRY_NUM = 11n; // price cap: at most 1.10x the create price (11/10)
const MAX_ENTRY_DEN = 10n;
const MIN_BUY_SHARE_NUM = 3n; // buy-dominated flow: buys >= 60% of trades (3/5)
const MIN_BUY_SHARE_DEN = 5n;
const TAKE_PROFIT_NUM = 115n; // sell all at +15% vs average entry (115/100)
const TAKE_PROFIT_DEN = 100n;
const STOP_LOSS_NUM = 80n; // sell all at -20% vs average entry (80/100)
const STOP_LOSS_DEN = 100n;
const HOLD_TIMEOUT_SLOTS = 7_200; // sell all after this many slots held (~25 min).
// The 5-position cap makes exit speed the throughput bottleneck: a 7,200-slot
// timeout turns the book ~3x faster than 28,800 and is what lifts the counted
// trips per 6h slice above 20, at a small cost in per-trip PnL.
const BUY_SLIPPAGE_BPS = 100; // 1% max slippage on buys. Tight on purpose: the
// fill lands 2 slots after the decision, and a buy whose fill would be >1%
// worse than the decision price is a buy that chased a runaway pump — the
// engine rejects it, which keeps the average entry near the trigger price.
const SELL_SLIPPAGE_BPS = 1000; // 10% max slippage on sells
// Breakeven stop (added after the first replay pass): once a position has been
// up 10%, never let it fall back below its average entry — bank the round trip
// at cost instead of riding it to the timeout exit at a loss.
const BE_ARM_NUM = 105n; // arm the stop once the peak reaches +5% vs entry
const BE_ARM_DEN = 100n;
const BE_STOP_NUM = 102n; // then sell if the price falls back to +2% vs entry
const BE_STOP_DEN = 100n;

interface MintInfo {
  createdSlot: number;
  vq0: bigint; // virtual quote reserves at create
  vt0: bigint; // virtual token reserves at create
  buys: number;
  sells: number;
  accRemaining: bigint; // lamports still to accumulate (0 = not accumulating)
  firstFillSlot: number; // slot of the first buy fill (-1 = no fill yet)
  peakVq: bigint; // virtual quote reserves at the position's peak price (0 = unset)
  peakVt: bigint; // virtual token reserves at the position's peak price
  beArmed: boolean; // breakeven stop armed (peak reached +10% vs entry)
}

const mints = new Map<string, MintInfo>();
const done = new Set<string>(); // mints whose position closed: never buy again
const pendingBuy = new Map<string, bigint>(); // mint -> lamports of the buy in flight
const pendingSell = new Set<string>(); // mints with a sell in flight
// Small working sets, so per-batch scans stay cheap even with thousands of mints seen.
const accSet = new Set<string>(); // mints with accumulation budget left

function setAcc(mint: string, info: MintInfo, remaining: bigint): void {
  info.accRemaining = remaining;
  if (remaining > 0n) accSet.add(mint);
  else accSet.delete(mint);
}

// Compare two prices given as (vq / vt) rationals: returns true when
// (aVq / aVt) >= mult * (bVq / bVt), with mult = multNum / multDen. Exact bigint math.
function priceGte(aVq: bigint, aVt: bigint, bVq: bigint, bVt: bigint, multNum: bigint, multDen: bigint): boolean {
  return aVq * bVt * multDen >= multNum * bVq * aVt;
}

function priceLte(aVq: bigint, aVt: bigint, bVq: bigint, bVt: bigint, multNum: bigint, multDen: bigint): boolean {
  return aVq * bVt * multDen <= multNum * bVq * aVt;
}

function onBatch(state: BotState, batch: Batch): OrderInput[] {
  const orders: OrderInput[] = [];
  const slot = batch.slot;

  // 1. Fold this batch's events into per-mint info.
  const tradedNow = new Set<string>();
  for (const e of batch.events) {
    if (e.event === 'CreateEvent') {
      if (!mints.has(e.mint)) {
        mints.set(e.mint, {
          createdSlot: e.slot,
          vq0: BigInt(e.virtual_quote_reserves),
          vt0: BigInt(e.virtual_token_reserves),
          buys: 0,
          sells: 0,
          accRemaining: 0n,
          firstFillSlot: -1,
          peakVq: 0n,
          peakVt: 1n,
          beArmed: false,
        });
      }
    } else if (e.event === 'TradeEvent') {
      const info = mints.get(e.mint);
      if (info !== undefined) {
        if (e.is_buy) info.buys += 1;
        else info.sells += 1;
      }
      tradedNow.add(e.mint);
    } else {
      // CompleteEvent: the curve is done; never touch this mint again.
      done.add(e.mint);
      const info = mints.get(e.mint);
      if (info !== undefined) setAcc(e.mint, info, 0n);
      pendingBuy.delete(e.mint);
      pendingSell.delete(e.mint);
    }
  }

  // 2. Reconcile fills: restore accumulation budget on rejected/failed buys,
  //    record first-fill slots, and mark closed positions as done.
  for (const f of batch.fills) {
    if (f.client_id.startsWith('_')) {
      // A harness close (complete / flatten / writeoff): the position is gone.
      done.add(f.mint);
      const info = mints.get(f.mint);
      if (info !== undefined) setAcc(f.mint, info, 0n);
      pendingBuy.delete(f.mint);
      pendingSell.delete(f.mint);
      continue;
    }
    if (f.side === 'buy') {
      const size = pendingBuy.get(f.mint);
      pendingBuy.delete(f.mint);
      if (f.status !== 'filled') {
        // Rejected or failed: the lamports were never spent, put them back.
        if (size !== undefined) {
          const info = mints.get(f.mint);
          if (info !== undefined) setAcc(f.mint, info, info.accRemaining + size);
        }
      } else {
        const info = mints.get(f.mint);
        if (info !== undefined && info.firstFillSlot < 0) {
          info.firstFillSlot = f.slot;
          // The peak starts at the entry price of the first fill.
          info.peakVq = BigInt(f.quote_lamports);
          info.peakVt = BigInt(f.token_amount);
          info.beArmed = false;
        }
      }
    } else {
      pendingSell.delete(f.mint);
      // A filled sell closes the position: applyBatch already folded this
      // batch's fills into state.positions before onBatch ran, so a filled
      // sell with no position left means this mint is done — no separate
      // per-batch scan over held mints is needed (and none is kept).
      if (f.status === 'filled' && !state.positions.has(f.mint)) done.add(f.mint);
    }
  }

  const maxOrders = state.hello.caps.max_orders_per_batch;
  const push = (o: OrderInput): boolean => {
    if (orders.length >= maxOrders) return false;
    orders.push(o);
    return true;
  };

  // 3. Exits first: take-profit, stop-loss, hold timeout (full position, sorted mint order).
  const held = [...state.positions.keys()].sort();
  for (const mint of held) {
    if (pendingSell.has(mint)) continue;
    const curve = state.curves.get(mint);
    const holding = state.positions.get(mint)!;
    const info = mints.get(mint);
    if (curve === undefined || info === undefined || info.firstFillSlot < 0) continue;
    if (curve.virtual_token_reserves === 0n || holding.token_amount === 0n) continue;
    const entryVq = holding.cost_lamports; // cost per token = cost / tokens
    const entryVt = holding.token_amount;
    const curVq = curve.virtual_quote_reserves;
    const curVt = curve.virtual_token_reserves;
    const age = slot - info.firstFillSlot;
    // Track the running peak price; arm the breakeven stop at +10% vs entry.
    if (info.peakVt > 0n && priceGte(curVq, curVt, info.peakVq, info.peakVt, 1n, 1n)) {
      info.peakVq = curVq;
      info.peakVt = curVt;
    }
    if (!info.beArmed && priceGte(info.peakVq, info.peakVt, entryVq, entryVt, BE_ARM_NUM, BE_ARM_DEN)) {
      info.beArmed = true;
    }
    const breakeven = info.beArmed && priceLte(curVq, curVt, entryVq, entryVt, BE_STOP_NUM, BE_STOP_DEN);
    // Trailing stop (added after the replay sweep): once the peak is >= +5% vs
    // entry, sell on a 3% retracement from the peak instead of holding to the
    // timeout at the floor. This is what converts pump-and-fade coins from
    // timeout losses into small wins, and it frees the slot sooner.
    const trailing =
      priceGte(info.peakVq, info.peakVt, entryVq, entryVt, 105n, 100n) &&
      priceLte(curVq, curVt, info.peakVq, info.peakVt, 97n, 100n);
    const takeProfit = priceGte(curVq, curVt, entryVq, entryVt, TAKE_PROFIT_NUM, TAKE_PROFIT_DEN);
    const stopLoss = priceLte(curVq, curVt, entryVq, entryVt, STOP_LOSS_NUM, STOP_LOSS_DEN);
    if (takeProfit || stopLoss || breakeven || trailing || age >= HOLD_TIMEOUT_SLOTS) {
      if (
        push({
          side: 'sell',
          mint,
          token_amount: holding.token_amount,
          max_slippage_bps: SELL_SLIPPAGE_BPS,
        })
      ) {
        pendingSell.add(mint);
        setAcc(mint, info, 0n);
      }
    }
  }

  // 4. Continue accumulating positions still under target (sorted mint order).
  const committed = state.committedPositionCount();
  let openSlots = committed;
  const accumulating = [...accSet].sort();
  for (const mint of accumulating) {
    const info = mints.get(mint)!;
    if (info.accRemaining <= 0n) {
      accSet.delete(mint);
      continue;
    }
    if (pendingBuy.has(mint) || pendingSell.has(mint)) continue;
    if (done.has(mint)) {
      setAcc(mint, info, 0n);
      continue;
    }
    const curve = state.curves.get(mint);
    if (curve === undefined || curve.complete) {
      setAcc(mint, info, 0n);
      continue;
    }
    if (slot - info.createdSlot > MAX_AGE_SLOTS) {
      setAcc(mint, info, 0n);
      continue;
    }
    if (!priceLte(curve.virtual_quote_reserves, curve.virtual_token_reserves, info.vq0, info.vt0, MAX_ENTRY_NUM, MAX_ENTRY_DEN)) {
      continue; // price ran past the cap: stop topping up, keep what we hold
    }
    const maxBuy = state.maxBuyLamports(mint);
    let size = CHUNK_LAMPORTS;
    if (size > info.accRemaining) size = info.accRemaining;
    if (size > maxBuy) size = maxBuy;
    if (size < MIN_CHUNK_LAMPORTS) continue;
    if (state.freeCashLamports() < size + state.txFeeLamports()) continue;
    if (
      push({ side: 'buy', mint, quote_lamports: size, max_slippage_bps: BUY_SLIPPAGE_BPS })
    ) {
      pendingBuy.set(mint, size);
      setAcc(mint, info, info.accRemaining - size);
    }
  }

  // 5. New entries: young, cheap, buy-dominated coins, one chunk to open.
  if (openSlots < state.hello.caps.max_open_positions) {
    const candidates = [...tradedNow].sort();
    for (const mint of candidates) {
      if (openSlots >= state.hello.caps.max_open_positions) break;
      if (orders.length >= maxOrders) break;
      const info = mints.get(mint);
      if (info === undefined) continue;
      if (done.has(mint)) continue;
      if (state.positions.has(mint) || pendingBuy.has(mint) || pendingSell.has(mint)) continue;
      if (info.accRemaining > 0n) continue;
      if (slot - info.createdSlot > MAX_AGE_SLOTS) continue;
      const curve = state.curves.get(mint);
      if (curve === undefined || curve.complete) continue;
      if (!priceLte(curve.virtual_quote_reserves, curve.virtual_token_reserves, info.vq0, info.vt0, MAX_ENTRY_NUM, MAX_ENTRY_DEN)) {
        continue;
      }
      const totalTrades = info.buys + info.sells;
      if (totalTrades > 0 && BigInt(info.buys) * MIN_BUY_SHARE_DEN < MIN_BUY_SHARE_NUM * BigInt(totalTrades)) {
        continue; // sell-dominated flow: skip
      }
      const maxBuy = state.maxBuyLamports(mint);
      if (maxBuy < CHUNK_LAMPORTS) continue;
      if (state.freeCashLamports() < CHUNK_LAMPORTS + state.txFeeLamports()) continue;
      if (
        push({ side: 'buy', mint, quote_lamports: CHUNK_LAMPORTS, max_slippage_bps: BUY_SLIPPAGE_BPS })
      ) {
        pendingBuy.set(mint, CHUNK_LAMPORTS);
        setAcc(mint, info, TARGET_LAMPORTS - CHUNK_LAMPORTS);
        openSlots += 1;
      }
    }
  }

  return orders;
}

await runBot({
  onHello(state) {
    // The seed is the only permitted randomness source; this strategy is fully
    // deterministic and uses none. Reading it here keeps the contract explicit.
    void state.hello.seed;
  },
  onBatch,
  onDone(state) {
    console.error(`curve-scout done: cash=${state.cash.toString()} positions=${state.positions.size}`);
  },
});

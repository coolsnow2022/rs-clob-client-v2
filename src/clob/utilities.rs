//! Client-side utility functions for orderbook analysis, fee calculation, and price validation.

use std::fmt::Write as _;

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use sha1::Digest as _;

use super::types::response::{OrderBookSummaryResponse, OrderSummary};
use super::types::{Amount, AmountInner, OrderType, Side, TickSize};
use crate::Result;
use crate::error::Error;

/// Number of decimal places in a USDC amount on-chain. Exposed so utility callers can
/// use the same truncation semantics.
pub const USDC_DECIMALS: u32 = 6;

/// Walks orderbook levels in reverse (worst-to-best), accumulating via `accumulate`,
/// and returns the cutoff price where cumulative ≥ `target`.
///
/// If no level satisfies the target:
/// - Returns `None` for [`OrderType::FOK`]
/// - Returns the best available price (first level) for other order types
///
/// Returns `None` for empty `levels`.
pub(crate) fn walk_levels<F: Fn(&OrderSummary) -> Decimal>(
    levels: &[OrderSummary],
    target: Decimal,
    accumulate: F,
    order_type: &OrderType,
) -> Option<Decimal> {
    if levels.is_empty() {
        return None;
    }

    let mut total = Decimal::ZERO;
    for level in levels.iter().rev() {
        total += accumulate(level);
        if total >= target {
            return Some(level.price);
        }
    }

    if *order_type == OrderType::FOK {
        return None;
    }

    Some(levels[0].price)
}

/// Walks the orderbook to calculate the effective fill price for a given [`Amount`].
///
/// The unit of `amount` (USDC vs shares) determines which side of the book is walked
/// and how liquidity is accumulated:
///
/// | Side | Amount  | Walks | Accumulates      |
/// |------|---------|-------|------------------|
/// | Buy  | Usdc    | asks  | `size * price`   |
/// | Buy  | Shares  | asks  | `size`           |
/// | Sell | Shares  | bids  | `size`           |
/// | Sell | Usdc    | — invalid, returns a validation error     |
///
/// # Errors
/// - `Side::Sell` paired with an `Amount::usdc(_)` (SELL orders must size in shares).
/// - `Side::Unknown`.
/// - `OrderType::FOK` with insufficient liquidity at any level.
///
/// For non-FOK order types with insufficient liquidity, returns the best available price.
pub fn calculate_market_price(
    orderbook: &OrderBookSummaryResponse,
    side: Side,
    amount: Amount,
    order_type: &OrderType,
) -> Result<Decimal> {
    let (levels, acc): (&[OrderSummary], fn(&OrderSummary) -> Decimal) = match (side, amount.0) {
        (Side::Buy, AmountInner::Usdc(_)) => (&orderbook.asks, |l| l.size * l.price),
        (Side::Buy, AmountInner::Shares(_)) => (&orderbook.asks, |l| l.size),
        (Side::Sell, AmountInner::Shares(_)) => (&orderbook.bids, |l| l.size),
        (Side::Sell, AmountInner::Usdc(_)) => {
            return Err(Error::validation(
                "SELL orders must specify their amount in shares, not USDC",
            ));
        }
        (Side::Unknown, _) => return Err(Error::validation(format!("Invalid side: {side}"))),
    };

    walk_levels(levels, amount.as_inner(), acc, order_type).ok_or_else(|| {
        Error::validation(format!(
            "Insufficient liquidity to fill {} on {side:?}",
            amount.as_inner()
        ))
    })
}

/// Generates a server-compatible SHA1 hash of an orderbook snapshot.
///
/// Constructs a compact JSON payload with a specific key order
/// (`market`, `asset_id`, `timestamp`, `hash=""`, `bids`, `asks`,
/// `min_order_size`, `tick_size`, `neg_risk`, `last_trade_price`)
/// and returns the SHA1 hex digest.
///
/// **Note**: [`OrderBookSummaryResponse::hash()`] uses SHA-256 on `serde_json::to_string`
/// and produces different results. This function is for server-compatible verification.
#[must_use]
pub fn orderbook_summary_hash(orderbook: &OrderBookSummaryResponse) -> String {
    // Build JSON manually — serde_json::json! uses BTreeMap which sorts keys alphabetically,
    // but the server expects a specific non-alphabetical key order.
    let mut json = String::with_capacity(512);

    json.push('{');
    let _ = write!(json, "\"market\":\"{}\"", orderbook.market);

    let asset_id_json = serde_json::to_string(&orderbook.asset_id).unwrap_or_default();
    let _ = write!(json, ",\"asset_id\":{asset_id_json}");
    let _ = write!(
        json,
        ",\"timestamp\":\"{}\"",
        orderbook.timestamp.timestamp_millis()
    );
    json.push_str(",\"hash\":\"\"");

    json.push_str(",\"bids\":[");
    for (i, o) in orderbook.bids.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        let _ = write!(
            json,
            "{{\"price\":\"{}\",\"size\":\"{}\"}}",
            o.price, o.size
        );
    }
    json.push(']');

    json.push_str(",\"asks\":[");
    for (i, o) in orderbook.asks.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        let _ = write!(
            json,
            "{{\"price\":\"{}\",\"size\":\"{}\"}}",
            o.price, o.size
        );
    }
    json.push(']');

    let _ = write!(json, ",\"min_order_size\":\"{}\"", orderbook.min_order_size);
    let _ = write!(
        json,
        ",\"tick_size\":\"{}\"",
        Decimal::from(orderbook.tick_size)
    );
    let _ = write!(json, ",\"neg_risk\":{}", orderbook.neg_risk);
    let last = orderbook.last_trade_price.unwrap_or(Decimal::ZERO);
    let _ = write!(json, ",\"last_trade_price\":\"{last}\"");
    json.push('}');

    let mut hasher = sha1::Sha1::new();
    hasher.update(json.as_bytes());
    let result = hasher.finalize();

    format!("{result:x}")
}

/// Adjusts a market-buy USDC amount to account for platform and builder taker fees.
///
/// Returns `amount` unchanged when `user_usdc_balance` already covers the total cost.
/// Otherwise shrinks it so principal + fees = balance, then truncates to [`USDC_DECIMALS`]
/// (matching the on-chain USDC scale). Returned amount is ready to pass to
/// [`Amount::usdc`](super::types::Amount::usdc).
///
/// # Errors
/// - `user_usdc_balance` is below the minimum to cover one USDC-unit of fees; the adjusted
///   amount would truncate to zero, which would submit a zero-value order the backend
///   rejects with an opaque error. Callers should top up the balance and retry.
pub fn adjust_market_buy_amount(
    amount: Decimal,
    user_usdc_balance: Decimal,
    price: Decimal,
    fee_rate: Decimal,
    fee_exponent: Decimal,
    builder_taker_fee_rate: Decimal,
) -> Result<Decimal> {
    let base = price * (Decimal::ONE - price);
    let base_f64: f64 = base.try_into().unwrap_or(0.0);
    let exp_f64: f64 = fee_exponent.try_into().unwrap_or(0.0);
    let platform_fee_rate =
        fee_rate * Decimal::try_from(base_f64.powf(exp_f64)).unwrap_or(Decimal::ZERO);

    let platform_fee = amount / price * platform_fee_rate;
    let total_cost = amount + platform_fee + amount * builder_taker_fee_rate;

    let raw = if user_usdc_balance < total_cost {
        let divisor = Decimal::ONE + platform_fee_rate / price + builder_taker_fee_rate;
        user_usdc_balance / divisor
    } else {
        amount
    };

    let adjusted = raw.trunc_with_scale(USDC_DECIMALS);
    if adjusted.is_zero() {
        return Err(Error::validation(format!(
            "user_usdc_balance {user_usdc_balance} too small to cover fees at price {price}; \
             fee-adjusted amount truncated to zero"
        )));
    }
    Ok(adjusted)
}

/// Validates that a price is within the valid range `[tick_size, 1 - tick_size]`.
#[must_use]
pub fn price_valid(price: Decimal, tick_size: TickSize) -> bool {
    let ts = Decimal::from(tick_size);
    price >= ts && price <= dec!(1) - ts
}

#[cfg(test)]
mod tests {
    use chrono::{DateTime, Utc};
    use rust_decimal_macros::dec;

    use super::*;
    use crate::types::{B256, U256};

    fn make_orderbook(
        bids: Vec<OrderSummary>,
        asks: Vec<OrderSummary>,
    ) -> OrderBookSummaryResponse {
        OrderBookSummaryResponse::builder()
            .market(B256::ZERO)
            .asset_id(U256::ZERO)
            .timestamp(Utc::now())
            .bids(bids)
            .asks(asks)
            .min_order_size(dec!(0.01))
            .neg_risk(false)
            .tick_size(TickSize::Hundredth)
            .build()
    }

    fn order(price: Decimal, size: Decimal) -> OrderSummary {
        OrderSummary::builder().price(price).size(size).build()
    }

    #[test]
    fn calculate_market_price_buy_usdc_sufficient_liquidity() {
        let ob = make_orderbook(
            vec![],
            vec![
                order(dec!(0.50), dec!(100)),
                order(dec!(0.51), dec!(100)),
                order(dec!(0.52), dec!(100)),
            ],
        );
        // Reversed walk: 0.52*100=52, 0.51*100=51, total=103 >= 80
        let amt = Amount::usdc(dec!(80)).unwrap();
        assert_eq!(
            calculate_market_price(&ob, Side::Buy, amt, &OrderType::FOK).unwrap(),
            dec!(0.51),
        );
    }

    #[test]
    fn calculate_market_price_buy_shares_sufficient_liquidity() {
        let ob = make_orderbook(
            vec![],
            vec![
                order(dec!(0.50), dec!(100)),
                order(dec!(0.51), dec!(100)),
                order(dec!(0.52), dec!(100)),
            ],
        );
        // Reversed walk (shares): 100, then 200 >= 150 → 0.51
        let amt = Amount::shares(dec!(150)).unwrap();
        assert_eq!(
            calculate_market_price(&ob, Side::Buy, amt, &OrderType::FOK).unwrap(),
            dec!(0.51),
        );
    }

    #[test]
    fn calculate_market_price_buy_insufficient_fok() {
        let ob = make_orderbook(vec![], vec![order(dec!(0.50), dec!(10))]);
        let amt = Amount::usdc(dec!(100)).unwrap();
        calculate_market_price(&ob, Side::Buy, amt, &OrderType::FOK).unwrap_err();
    }

    #[test]
    fn calculate_market_price_buy_insufficient_fak() {
        let ob = make_orderbook(
            vec![],
            vec![order(dec!(0.50), dec!(10)), order(dec!(0.60), dec!(5))],
        );
        let amt = Amount::usdc(dec!(1000)).unwrap();
        assert_eq!(
            calculate_market_price(&ob, Side::Buy, amt, &OrderType::FAK).unwrap(),
            dec!(0.50),
        );
    }

    #[test]
    fn calculate_market_price_sell_shares() {
        let ob = make_orderbook(
            vec![
                order(dec!(0.50), dec!(100)),
                order(dec!(0.49), dec!(100)),
                order(dec!(0.48), dec!(100)),
            ],
            vec![],
        );
        // Reversed walk: 0.48 (100), 0.49 (200), need 150 tokens
        let amt = Amount::shares(dec!(150)).unwrap();
        assert_eq!(
            calculate_market_price(&ob, Side::Sell, amt, &OrderType::FOK).unwrap(),
            dec!(0.49),
        );
    }

    #[test]
    fn calculate_market_price_sell_usdc_is_rejected() {
        let ob = make_orderbook(
            vec![order(dec!(0.49), dec!(100))],
            vec![order(dec!(0.51), dec!(100))],
        );
        let amt = Amount::usdc(dec!(10)).unwrap();
        calculate_market_price(&ob, Side::Sell, amt, &OrderType::FOK).unwrap_err();
    }

    #[test]
    fn calculate_market_price_empty_orderbook() {
        let ob = make_orderbook(vec![], vec![]);
        let amt = Amount::usdc(dec!(100)).unwrap();
        calculate_market_price(&ob, Side::Buy, amt, &OrderType::FOK).unwrap_err();
    }

    #[test]
    fn calculate_market_price_unknown_side_errors() {
        let ob = make_orderbook(
            vec![order(dec!(0.49), dec!(100))],
            vec![order(dec!(0.51), dec!(100))],
        );
        let amt = Amount::usdc(dec!(10)).unwrap();
        calculate_market_price(&ob, Side::Unknown, amt, &OrderType::FOK).unwrap_err();
    }

    #[test]
    fn price_valid_within_bounds() {
        assert!(price_valid(dec!(0.5), TickSize::Hundredth));
        assert!(price_valid(dec!(0.01), TickSize::Hundredth));
        assert!(price_valid(dec!(0.99), TickSize::Hundredth));
    }

    #[test]
    fn price_valid_at_boundaries() {
        assert!(price_valid(dec!(0.1), TickSize::Tenth));
        assert!(price_valid(dec!(0.9), TickSize::Tenth));
    }

    #[test]
    fn price_valid_out_of_bounds() {
        assert!(!price_valid(dec!(0.0), TickSize::Hundredth));
        assert!(!price_valid(dec!(1.0), TickSize::Hundredth));
        assert!(!price_valid(dec!(0.005), TickSize::Hundredth));
        assert!(!price_valid(dec!(0.995), TickSize::Hundredth));
    }

    #[test]
    fn price_valid_all_tick_sizes() {
        assert!(price_valid(dec!(0.5), TickSize::Tenth));
        assert!(price_valid(dec!(0.5), TickSize::Hundredth));
        assert!(price_valid(dec!(0.5), TickSize::Thousandth));
        assert!(price_valid(dec!(0.5), TickSize::TenThousandth));
    }

    #[test]
    fn orderbook_hash_deterministic() {
        let ts = DateTime::from_timestamp_millis(1_700_000_000_000).expect("valid ts");
        let ob = OrderBookSummaryResponse::builder()
            .market(B256::ZERO)
            .asset_id(U256::ZERO)
            .timestamp(ts)
            .bids(vec![order(dec!(0.49), dec!(50))])
            .asks(vec![order(dec!(0.51), dec!(25))])
            .min_order_size(dec!(0.01))
            .neg_risk(false)
            .tick_size(TickSize::Hundredth)
            .build();

        let hash = orderbook_summary_hash(&ob);
        assert_eq!(hash.len(), 40);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(hash, orderbook_summary_hash(&ob));
    }

    #[test]
    fn orderbook_hash_differs_for_different_inputs() {
        let ts = DateTime::from_timestamp_millis(1_700_000_000_000).expect("valid ts");
        let ob1 = OrderBookSummaryResponse::builder()
            .market(B256::ZERO)
            .asset_id(U256::from(1_u64))
            .timestamp(ts)
            .min_order_size(dec!(0.01))
            .neg_risk(false)
            .tick_size(TickSize::Hundredth)
            .build();

        let ob2 = OrderBookSummaryResponse::builder()
            .market(B256::ZERO)
            .asset_id(U256::from(2_u64))
            .timestamp(ts)
            .min_order_size(dec!(0.01))
            .neg_risk(false)
            .tick_size(TickSize::Hundredth)
            .build();

        assert_ne!(orderbook_summary_hash(&ob1), orderbook_summary_hash(&ob2));
    }

    #[test]
    fn adjust_market_buy_no_adjustment_when_balance_sufficient() {
        let result = adjust_market_buy_amount(
            dec!(100),
            dec!(1000),
            dec!(0.5),
            dec!(0.02),
            dec!(1),
            dec!(0),
        )
        .unwrap();
        assert_eq!(result, dec!(100));
    }

    #[test]
    fn adjust_market_buy_adjusts_when_balance_insufficient() {
        let result = adjust_market_buy_amount(
            dec!(100),
            dec!(100),
            dec!(0.5),
            dec!(0.02),
            dec!(1),
            dec!(0),
        )
        .unwrap();
        assert!(result < dec!(100));
        assert!(result > dec!(0));
    }

    #[test]
    fn adjust_market_buy_with_builder_fee() {
        let result = adjust_market_buy_amount(
            dec!(100),
            dec!(100),
            dec!(0.5),
            dec!(0),
            dec!(1),
            dec!(0.005),
        )
        .unwrap();
        // effective * 1.005 = 100, truncated to 6 USDC decimals.
        let expected = (dec!(100) / dec!(1.005)).trunc_with_scale(USDC_DECIMALS);
        assert_eq!(result, expected);
    }

    #[test]
    fn adjust_market_buy_errors_when_balance_truncates_to_zero() {
        // user_usdc_balance smaller than 1e-6 after fee-divisor → truncates to zero.
        let err = adjust_market_buy_amount(
            dec!(100),       // wanted amount
            dec!(0.0000001), // balance well below 1 USDC-micro
            dec!(0.5),
            dec!(0.02),
            dec!(1),
            dec!(0.005),
        )
        .unwrap_err();
        assert!(err.to_string().contains("truncated to zero"));
    }
}

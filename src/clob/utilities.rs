//! Client-side utility functions for orderbook analysis, fee calculation, and price validation.
//!
//! These utilities match the behavior of the Python CLOB client v2's `utilities.py` and
//! `order_builder/builder.py` for feature parity.

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use sha1::Digest as _;

use super::types::response::{OrderBookSummaryResponse, OrderSummary};
use super::types::{OrderType, Side, TickSize};

/// Walks the orderbook to calculate the effective fill price for a given amount.
///
/// Matches the Python client's `calculate_buy_market_price` / `calculate_sell_market_price`:
/// - Iterates positions in reverse order (worst-to-best price levels)
/// - For BUY: accumulates cumulative USDC cost (`size * price`) until `>= amount`
/// - For SELL: accumulates cumulative token size until `>= amount`
/// - For [`OrderType::FOK`]: returns `None` if insufficient liquidity
/// - For other order types: returns the first available price if any liquidity exists
#[must_use]
pub fn calculate_market_price(
    orderbook: &OrderBookSummaryResponse,
    side: Side,
    amount: Decimal,
    order_type: &OrderType,
) -> Option<Decimal> {
    let positions: &[OrderSummary] = match side {
        Side::Buy => &orderbook.asks,
        Side::Sell | Side::Unknown => &orderbook.bids,
    };

    if positions.is_empty() {
        return None;
    }

    let mut total = Decimal::ZERO;
    for p in positions.iter().rev() {
        match side {
            Side::Buy => total += p.size * p.price,
            Side::Sell | Side::Unknown => total += p.size,
        }
        if total >= amount {
            return Some(p.price);
        }
    }

    // Insufficient liquidity to fill the full amount
    if *order_type == OrderType::FOK {
        return None;
    }

    // For non-FOK orders, return the best available price (first position)
    Some(positions[0].price)
}

/// Generates a server-compatible SHA1 hash of an orderbook snapshot.
///
/// Algorithm (matches Python's `generate_orderbook_summary_hash` exactly):
/// 1. Construct a JSON object with keys in this exact order:
///    `market`, `asset_id`, `timestamp`, `hash` (empty string), `bids`, `asks`,
///    `min_order_size`, `tick_size`, `neg_risk`, `last_trade_price`
/// 2. Serialize with compact separators (no spaces)
/// 3. SHA1 hash the serialized string
/// 4. Return the hex digest
///
/// **Note**: The existing [`OrderBookSummaryResponse::hash()`] method uses SHA-256 on
/// `serde_json::to_string` and produces different results. This function is for
/// server-compatible verification.
#[must_use]
pub fn orderbook_summary_hash(orderbook: &OrderBookSummaryResponse) -> String {
    let bids: Vec<serde_json::Value> = orderbook
        .bids
        .iter()
        .map(|o| serde_json::json!({"price": o.price, "size": o.size}))
        .collect();

    let asks: Vec<serde_json::Value> = orderbook
        .asks
        .iter()
        .map(|o| serde_json::json!({"price": o.price, "size": o.size}))
        .collect();

    // Key order must match Python exactly
    let payload = serde_json::json!({
        "market": orderbook.market,
        "asset_id": orderbook.asset_id,
        "timestamp": orderbook.timestamp.timestamp_millis().to_string(),
        "hash": "",
        "bids": bids,
        "asks": asks,
        "min_order_size": orderbook.min_order_size,
        "tick_size": Decimal::from(orderbook.tick_size),
        "neg_risk": orderbook.neg_risk,
        "last_trade_price": orderbook.last_trade_price.unwrap_or(Decimal::ZERO),
    });

    let serialized = serde_json::to_string(&payload).unwrap_or_default();

    let mut hasher = sha1::Sha1::new();
    hasher.update(serialized.as_bytes());
    let result = hasher.finalize();

    format!("{result:x}")
}

/// Adjusts a market buy USDC amount to account for platform and builder fees.
///
/// Only adjusts when `user_usdc_balance <= total_cost` (the cost of buying `amount` worth
/// of tokens including all fees). Returns the effective amount that can be traded after fees.
///
/// Matches Python's `adjust_market_buy_amount` from `utilities.py`.
///
/// # Arguments
///
/// * `amount` - The desired USDC amount to spend
/// * `user_usdc_balance` - The user's available USDC balance
/// * `price` - The market price of the token
/// * `fee_rate` - Platform fee rate (as decimal, e.g. `0.02` for 2%)
/// * `fee_exponent` - Platform fee exponent for the fee formula
/// * `builder_taker_fee_rate` - Builder taker fee rate (as decimal, `0` if no builder)
#[must_use]
pub fn adjust_market_buy_amount(
    amount: Decimal,
    user_usdc_balance: Decimal,
    price: Decimal,
    fee_rate: Decimal,
    fee_exponent: Decimal,
    builder_taker_fee_rate: Decimal,
) -> Decimal {
    // base = price * (1 - price)
    let base = price * (Decimal::ONE - price);
    // platform_fee_rate = fee_rate * (base ^ fee_exponent)
    // We need float for the exponentiation
    let base_f64: f64 = base.try_into().unwrap_or(0.0);
    let exp_f64: f64 = fee_exponent.try_into().unwrap_or(0.0);
    let pow_result = base_f64.powf(exp_f64);
    let platform_fee_rate = fee_rate * Decimal::try_from(pow_result).unwrap_or(Decimal::ZERO);

    // platform_fee = amount / price * platform_fee_rate
    let platform_fee = amount / price * platform_fee_rate;
    // total_cost = amount + platform_fee + amount * builder_taker_fee_rate
    let total_cost = amount + platform_fee + amount * builder_taker_fee_rate;

    if user_usdc_balance <= total_cost {
        // Adjust: balance / (1 + platform_fee_rate / price + builder_taker_fee_rate)
        let divisor = Decimal::ONE + platform_fee_rate / price + builder_taker_fee_rate;
        user_usdc_balance / divisor
    } else {
        amount
    }
}

/// Validates that a price is within the valid range `[tick_size, 1 - tick_size]`.
#[must_use]
pub fn price_valid(price: Decimal, tick_size: TickSize) -> bool {
    let ts = Decimal::from(tick_size);
    price >= ts && price <= dec!(1) - ts
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
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
    fn calculate_market_price_buy_sufficient_liquidity() {
        let ob = make_orderbook(
            vec![],
            vec![
                order(dec!(0.50), dec!(100)),
                order(dec!(0.51), dec!(100)),
                order(dec!(0.52), dec!(100)),
            ],
        );
        // Reversed walk: 0.52 (52), 0.51 (103), need 80 USDC
        // 0.52*100 = 52, 0.51*100 = 51, total = 103 >= 80
        let result = calculate_market_price(&ob, Side::Buy, dec!(80), &OrderType::FOK);
        assert_eq!(result, Some(dec!(0.51)));
    }

    #[test]
    fn calculate_market_price_buy_insufficient_fok() {
        let ob = make_orderbook(vec![], vec![order(dec!(0.50), dec!(10))]);
        // 0.50 * 10 = 5 USDC, need 100
        let result = calculate_market_price(&ob, Side::Buy, dec!(100), &OrderType::FOK);
        assert_eq!(result, None);
    }

    #[test]
    fn calculate_market_price_buy_insufficient_fak() {
        let ob = make_orderbook(
            vec![],
            vec![order(dec!(0.50), dec!(10)), order(dec!(0.60), dec!(5))],
        );
        // Not enough, but FAK returns first position price
        let result = calculate_market_price(&ob, Side::Buy, dec!(1000), &OrderType::FAK);
        assert_eq!(result, Some(dec!(0.50)));
    }

    #[test]
    fn calculate_market_price_sell() {
        let ob = make_orderbook(
            vec![
                order(dec!(0.50), dec!(100)),
                order(dec!(0.49), dec!(100)),
                order(dec!(0.48), dec!(100)),
            ],
            vec![],
        );
        // Reversed walk: 0.48 (100), 0.49 (200), need 150 tokens
        let result = calculate_market_price(&ob, Side::Sell, dec!(150), &OrderType::FOK);
        assert_eq!(result, Some(dec!(0.49)));
    }

    #[test]
    fn calculate_market_price_empty_orderbook() {
        let ob = make_orderbook(vec![], vec![]);
        let result = calculate_market_price(&ob, Side::Buy, dec!(100), &OrderType::FOK);
        assert_eq!(result, None);
    }

    #[test]
    fn price_valid_within_bounds() {
        assert!(price_valid(dec!(0.5), TickSize::Hundredth));
        assert!(price_valid(dec!(0.01), TickSize::Hundredth));
        assert!(price_valid(dec!(0.99), TickSize::Hundredth));
    }

    #[test]
    fn price_valid_at_boundaries() {
        // At exact boundary — should be valid (inclusive)
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
    fn adjust_market_buy_no_adjustment_when_balance_sufficient() {
        let result = adjust_market_buy_amount(
            dec!(100),  // amount
            dec!(1000), // balance (way more than needed)
            dec!(0.5),  // price
            dec!(0.02), // fee_rate
            dec!(1),    // fee_exponent
            dec!(0),    // builder fee
        );
        assert_eq!(result, dec!(100));
    }

    #[test]
    fn adjust_market_buy_adjusts_when_balance_insufficient() {
        let result = adjust_market_buy_amount(
            dec!(100),  // amount
            dec!(100),  // balance (equal to amount, but fees will push it over)
            dec!(0.5),  // price
            dec!(0.02), // fee_rate
            dec!(1),    // fee_exponent
            dec!(0),    // builder fee
        );
        // Should be less than 100 since fees eat into the balance
        assert!(result < dec!(100));
        assert!(result > dec!(0));
    }

    #[test]
    fn adjust_market_buy_with_builder_fee() {
        let result = adjust_market_buy_amount(
            dec!(100),   // amount
            dec!(100),   // balance
            dec!(0.5),   // price
            dec!(0),     // platform fee_rate
            dec!(1),     // fee_exponent
            dec!(0.005), // builder taker fee (0.5%)
        );
        // effective + effective * 0.005 = 100
        // effective * 1.005 = 100
        // effective = 100 / 1.005 ≈ 99.50248756...
        let expected = dec!(100) / dec!(1.005);
        assert_eq!(result, expected);
    }
}

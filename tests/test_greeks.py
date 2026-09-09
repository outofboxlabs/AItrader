from portfolio_monitor.greeks import black_scholes_greeks


def test_call_put_delta_parity():
    call = black_scholes_greeks(
        spot=100, strike=100, years_to_expiry=0.5, risk_free_rate=0.045, iv=0.30, option_type="call"
    )
    put = black_scholes_greeks(
        spot=100, strike=100, years_to_expiry=0.5, risk_free_rate=0.045, iv=0.30, option_type="put"
    )
    # put-call parity on delta: call_delta - put_delta == 1
    assert abs((call.delta - put.delta) - 1.0) < 1e-9
    # gamma and vega are identical for calls and puts at the same strike/expiry/iv
    assert abs(call.gamma - put.gamma) < 1e-9
    assert abs(call.vega - put.vega) < 1e-9


def test_atm_call_delta_near_half():
    call = black_scholes_greeks(
        spot=100, strike=100, years_to_expiry=0.25, risk_free_rate=0.045, iv=0.25, option_type="call"
    )
    assert 0.5 < call.delta < 0.65


def test_long_option_theta_is_negative():
    call = black_scholes_greeks(
        spot=100, strike=100, years_to_expiry=0.1, risk_free_rate=0.045, iv=0.30, option_type="call"
    )
    assert call.theta < 0


def test_degenerate_inputs_return_zero_greeks():
    expired = black_scholes_greeks(
        spot=100, strike=100, years_to_expiry=0.0, risk_free_rate=0.045, iv=0.30, option_type="call"
    )
    assert expired.delta == 0.0
    assert expired.gamma == 0.0
    assert expired.theta == 0.0
    assert expired.vega == 0.0

    zero_iv = black_scholes_greeks(
        spot=100, strike=100, years_to_expiry=0.5, risk_free_rate=0.045, iv=0.0, option_type="call"
    )
    assert zero_iv.delta == 0.0


def test_deep_itm_call_delta_near_one():
    call = black_scholes_greeks(
        spot=200, strike=100, years_to_expiry=0.5, risk_free_rate=0.045, iv=0.20, option_type="call"
    )
    assert call.delta > 0.95


def test_deep_otm_put_delta_near_zero():
    put = black_scholes_greeks(
        spot=200, strike=100, years_to_expiry=0.5, risk_free_rate=0.045, iv=0.20, option_type="put"
    )
    assert abs(put.delta) < 0.05

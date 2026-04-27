trading_policies = {
    'aggressive': {
        'long': {
            'trend_down': 99,
            'trend_up': 75,
            'range': 75,
            'transition_up': 90,
            'transition_down': 90,
            'breakout_wait_up': 75,
            'breakout_wait_down': 99,
            '_global': 99
        }, 'short': {
            'trend_down': 75,
            'trend_up': 99,
            'range': 75,
            'transition_up': 90,
            'transition_down': 90,
            'breakout_wait_up': 99,
            'breakout_wait_down': 75,
            '_global': 99
        },
    },
    'normal': {
        'long': {
            'trend_down': 99,
            'trend_up': 75,
            'range': 90,
            'transition_up': 90,
            'transition_down': 99,
            'breakout_wait_up': 90,
            'breakout_wait_down': 99,
            '_global': 99
        }, 'short': {
            'trend_down': 75,
            'trend_up': 99,
            'range': 90,
            'transition_up': 99,
            'transition_down': 90,
            'breakout_wait_up': 99,
            'breakout_wait_down': 90,
            '_global': 99
        }
    },
    'conservative': {
        'long': {
            'trend_down': 99,
            'trend_up': 90,
            'range': 95,
            'transition_up': 99,
            'transition_down': 99,
            'breakout_wait_up': 95,
            'breakout_wait_down': 99,
            '_global': 99
        }, 'short': {
            'trend_down': 90,
            'trend_up': 99,
            'range': 95,
            'transition_up': 99,
            'transition_down': 99,
            'breakout_wait_up': 99,
            'breakout_wait_down': 95,
            '_global': 99
        }
    }
}
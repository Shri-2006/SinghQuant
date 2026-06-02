import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from core.features import build_features
from data.sentiment_fetcher import add_sentiment_to_df

class TradingEnvironment(gym.Env):
    """
    Custom Gymnasium trading environment for risky2 RL bot.
    
    State: price features + current position + unrealized PnL
    Actions: 0=HOLD, 1=BUY, 2=SELL
    Reward: realized PnL with incentives to buy and hold winners,
            penalties for holding losers and missing opportunities
    """

    def __init__(self, df, initial_capital=200.0, max_position_pct=0.25):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.max_position = initial_capital * max_position_pct

        self.action_space = spaces.Discrete(3)

        n_features = len(self._get_feature_cols())
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features + 2,),
            dtype=np.float32
        )

        self.reset()

    def _get_feature_cols(self):
        """Returns feature columns — excludes raw OHLCV"""
        return [c for c in self.df.columns
                if c not in ['open', 'high', 'low', 'close', 'volume']]

    def reset(self, seed=None, options=None):
        """Reset environment to starting state for new episode"""
        super().reset(seed=seed)
        self.current_step = 0
        self.cash         = self.initial_capital
        self.position     = 0.0
        self.entry_price  = 0.0
        self.total_pnl    = 0.0
        self.peak_value   = self.initial_capital
        self.trade_count  = 0
        self.steps_in_position = 0  # track how long we hold
        return self._get_observation(), {}

    def _get_observation(self):
        """Build the state vector the agent sees"""
        row          = self.df.iloc[self.current_step]
        feature_cols = self._get_feature_cols()
        features     = row[feature_cols].values.astype(np.float32)

        position_norm  = self.position / self.initial_capital
        unrealized_pnl = 0.0
        if self.position > 0 and self.entry_price > 0:
            current_price  = float(row['close'])
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price

        return np.append(features, [position_norm, unrealized_pnl]).astype(np.float32)

    def step(self, action):
        """
        Execute one trading action and return new state, reward, done flag.
        
        Reward design:
        - BUY:  small positive reward to encourage exploration
        - SELL: realized PnL as reward (positive or negative)
        - HOLD with position: reward unrealized gains, penalize unrealized losses
        - HOLD without position when price is rising: small penalty for missing opportunity
        """
        row   = self.df.iloc[self.current_step]
        price = float(row['close'])
        reward = 0.0

        # Action 1: BUY
        if action == 1 and self.position == 0:
            self.position    = min(self.max_position, self.cash)
            self.cash       -= self.position
            self.entry_price = price
            self.trade_count += 1
            self.steps_in_position = 0
            # Small positive reward for taking a position — encourages exploration
            reward = 0.001

        # Action 2: SELL
        elif action == 2 and self.position > 0:
            pnl         = self.position * (price - self.entry_price) / self.entry_price
            self.cash  += self.position + pnl
            self.total_pnl += pnl
            # Normalize reward by initial capital
            reward      = pnl / self.initial_capital
            # Bonus for profitable trades to reinforce good behavior
            if pnl > 0:
                reward *= 1.5
            self.position    = 0.0
            self.entry_price = 0.0
            self.trade_count += 1
            self.steps_in_position = 0

        # Action 0: HOLD
        else:
            if self.position > 0 and self.entry_price > 0:
                self.steps_in_position += 1
                unrealized = self.position * (price - self.entry_price) / self.entry_price
                # Reward for holding winners, penalize holding losers
                if unrealized > 0:
                    reward = unrealized * 0.0005  # small positive for riding winners
                else:
                    reward = unrealized * 0.002   # stronger penalty for holding losers
            else:
                # Penalize holding cash when next bar is higher (missed opportunity)
                if self.current_step < len(self.df) - 2:
                    next_price = float(self.df.iloc[self.current_step + 1]['close'])
                    if next_price > price:
                        reward = -0.0005  # small penalty for missing upward move

        # Track peak value for drawdown
        portfolio_value = self.cash + self.position
        self.peak_value = max(self.peak_value, portfolio_value)
        drawdown        = (portfolio_value - self.peak_value) / self.peak_value

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        return self._get_observation(), reward, done, False, {
            "total_pnl": self.total_pnl,
            "drawdown":  drawdown,
            "trades":    self.trade_count
        }

    def render(self):
        """Simple text render for debugging during training"""
        portfolio_value = self.cash + self.position
        print(
            f"Step {self.current_step} | "
            f"Portfolio: ${portfolio_value:.2f} | "
            f"Cash: ${self.cash:.2f} | "
            f"Position: ${self.position:.2f} | "
            f"Total PnL: ${self.total_pnl:.2f}"
        )
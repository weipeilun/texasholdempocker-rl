import sys

from .constants import *


class DummyAgent(object):
    """
    Dummy agent is designed to easily interact with the
    game engine. The agent will first be told what action
    to perform. Then the environment will call this agent
    to perform the actual action. This can help us to
    isolate environment and agents towards a gym like
    interface.
    """
    def __init__(self, value_left):
        self.status = PlayerStatus.ONBOARD
        self.action = None
        self.value_left = value_left

        # 记录本局游戏下注和赢注，用于计算奖励
        self.value_game_start = self.value_left
        self.value_bet = 0
        self.value_win = 0
        self.game_result = GamePlayerResult.DEFAULT    # win/even/lose

    def act(self, action, current_round_min_value, raised_value):
        """
        Simply return the action that is set previously.
        """
        self.set_action(action, current_round_min_value, raised_value)

        return [*self.action, self.value_left]

    def determine_raise_call(self, bet_value, current_round_min_value, raised_value):
        """
        # PockerTDA.com规则[46.C]
        下注数额大于current_round_min_value的1.5倍，视作raise，否则视作call.
        """
        call_value_threshold = int(current_round_min_value * 1.5)
        if bet_value + raised_value < call_value_threshold:
            final_bet_value = min(current_round_min_value - raised_value, self.value_left)
            if final_bet_value <= 0:
                final_bet_action = PlayerActions.CHECK
            else:
                final_bet_action = PlayerActions.CALL
        else:
            final_bet_action = PlayerActions.RAISE
            final_bet_value = min(max(current_round_min_value * 2, bet_value), self.value_left)
        return final_bet_action, final_bet_value

    def set_action(self, action, current_round_min_value, raised_value):
        assert action is not None, f"Critical: action can not be None."
        assert self.status == PlayerStatus.ONBOARD, f'Only onboard player can take action, current player status:{self.status}'

        self.action = action

        # 实际的call阈值线，低于这个线的raise都要补到这个线，否则allin
        # 实际的raise阈值线是raise_min_value，raise不可低于这个
        if isinstance(action[0], int) and isinstance(action[1], float):
            # action_value(action[1])可能从模型出来，需要处理异常值
            action_value_modified = min(max(action[1], 0), 1.)
            pa = PlayerActions(action[0])
            if action_value_modified == 0:
                if pa == PlayerActions.RAISE or pa == PlayerActions.CALL:
                    pa = PlayerActions.CHECK

            # 注意此处bet_value已经定到了合法区间中
            bet_value = max(int(self.value_left * action_value_modified), 1)
            if pa == PlayerActions.FOLD:
                actual_action = (PlayerActions.FOLD, 0)
            elif pa == PlayerActions.CHECK:
                # PockerTDA.com规则[55]，声明check，可以call不能加注
                if current_round_min_value <= raised_value:
                    actual_action = (PlayerActions.CHECK, 0)
                else:
                    actual_action = self.determine_raise_call(0, current_round_min_value, raised_value)
            elif pa == PlayerActions.RAISE:
                actual_action = self.determine_raise_call(bet_value, current_round_min_value, raised_value)
            elif pa == PlayerActions.CALL:
                # PockerTDA.com规则[55]，声明call，可以视作check
                if current_round_min_value <= raised_value:
                    actual_action = (PlayerActions.CHECK, 0)
                else:
                    actual_action = self.determine_raise_call(bet_value, current_round_min_value, raised_value)
            else:
                raise ValueError(f'Unsupported action: {action}')
        elif isinstance(action[0], PlayerActions) and isinstance(action[1], int):
            actual_action = action
        else:
            raise ValueError(f'Unsupported action: {action}')

        self.action = actual_action
        if self.action[0] == PlayerActions.FOLD:
            self.status = PlayerStatus.FOLDED
        elif (self.action[0] == PlayerActions.CALL or self.action[0] == PlayerActions.RAISE) and self.action[1] >= self.value_left:
            self.action = (self.action[0], self.value_left)
            self.status = PlayerStatus.ALLIN

        value_bet = self.action[1]
        self.value_left -= value_bet
        self.value_bet += value_bet

    def reset_action(self):
        self.action = None

    def set_blinds(self, value):
        self.set_action((PlayerActions.RAISE, value), 0, 0)
        actual_bet = self.action[1]
        self.reset_action()
        return actual_bet

    def __str__(self):
        return f'value_left:{int(self.value_left)}, status:{self.status}, action:{self.action}'

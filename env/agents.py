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
    def __init__(self, value_left, small_bind):
        self.status = PlayerStatus.ONBOARD
        self.action = None
        self.value_left = value_left
        self.small_bind = small_bind

        # 记录本局游戏下注和赢注，用于计算奖励
        self.value_game_start = self.value_left
        self.value_bet = 0
        self.value_win = 0
        self.game_result = GamePlayerResult.DEFAULT    # win/even/lose

    def validate_action(self, action, current_round_min_value, current_round_raise_min_value, raised_value):
        assert action is not None, ValueError(f"Critical: action can not be None.")
        assert self.status == PlayerStatus.ONBOARD, ValueError(f'Only onboard player can take action, current player status:{self.status}')

        assert isinstance(action[0], PlayerActions) and isinstance(action[1], int), ValueError(f'Wrong action type: ({type(action[0])}, {type(action[1])})')

        if action[1] % self.small_bind != 0:
            raise ValueError(f'Action value is not a multiple of small bind ({self.small_bind}), which is {action[1]}')

        if action[0] in (PlayerActions.RAISE, PlayerActions.CHECK_CALL):
            assert 0 <= action[1] - raised_value <= self.value_left, ValueError(f'Try to raise a delta {action[1] - raised_value} but need to be no less than 0 and no more than {self.value_left}')

            if action[0] == PlayerActions.RAISE and current_round_min_value + current_round_raise_min_value > action[1]:
                raise ValueError(f'Try to raise {action[1]} but need to be at least {current_round_min_value + current_round_raise_min_value}')

            if action[0] == PlayerActions.CHECK_CALL and current_round_min_value > action[1]:
                raise ValueError(f'Try to call {action[1]} but need to be at least {current_round_min_value}')

    def act(self, action, current_round_min_value, current_round_raise_min_value, raised_value):
        """
        Simply return the action that is set previously.
        """
        self.validate_action(action, current_round_min_value, current_round_raise_min_value, raised_value)

        self.set_action(action, raised_value)

        return self.action

    def set_action(self, action, raised_value):
        actual_action, actual_action_value = action
        # 此处实际执行的action算增量
        actual_delta_action_value = max(actual_action_value - raised_value, 0)
        if actual_action == PlayerActions.CHECK_CALL:
            # call but no enough value left, treat as all in
            if actual_delta_action_value >= self.value_left:
                actual_delta_action_value = self.value_left
                actual_action_value = actual_delta_action_value + raised_value
                self.status = PlayerStatus.ALLIN
        elif actual_action == PlayerActions.RAISE:
            # raise but no enough value left, treat as all in
            if actual_delta_action_value >= self.value_left:
                actual_delta_action_value = self.value_left
                actual_action_value = actual_delta_action_value + raised_value
                self.status = PlayerStatus.ALLIN
        elif actual_action == PlayerActions.FOLD:
            actual_action_value = raised_value
            self.status = PlayerStatus.FOLDED

        self.action = (actual_action, actual_action_value, actual_delta_action_value)
        self.value_left -= actual_delta_action_value
        self.value_bet += actual_delta_action_value

    def reset_action(self):
        self.action = None

    def set_blinds(self, value):
        actual_bet = value
        if value >= self.value_left:
            actual_bet = self.value_left
            self.status = PlayerStatus.ALLIN
        self.value_left -= actual_bet
        self.value_bet += actual_bet
        return actual_bet

    def __str__(self):
        return f'value_left:{int(self.value_left)}, status:{self.status}, action:{self.action}'

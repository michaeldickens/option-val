from collections import namedtuple
from decimal import Decimal as D
from scipy import optimize
import datetime
import math
import csv

class I(namedtuple('IndexBase', 't n_up')):
    """Index into the lattice of option states.

    fields:
        t: total timesteps elapsed
        n_up: number of timesteps the valuation went up
    """
    def __init__(self, t, n_up):
        assert 0 <= self.n_up <= self.t, \
            "We want 0 <= %s <= %s" % (self.n_up, self.t)

    @property
    def n_down(self):
        return self.t - self.n_up

    def go_up(self):
        return I(self.t + 1, self.n_up + 1)

    def go_down(self):
        return I(self.t + 1, self.n_up)

    def go_back_up(self):
        return I(self.t - 1, self.n_up)

    def go_back_down(self):
        return I(self.t - 1, self.n_up - 1)


def memoize(fn):
    """Memoize a method of the Params subclass, for dynamic programming"""
    # inval the cache bc we're probably redefining a function
    def inner(self, *args):
        cache = self.cache.setdefault(fn.__name__, {})
        if args not in cache:
            cache[args] = fn(self, *args)
        return cache[args]
    return inner


class ModelBase:
    # company parameters
    initial_valuation = None
    strike_price = None          # strike price of an option on the entire company
    annual_volatility = 0.54     # 2x S&P 500 annual vol
    annual_growth = 1.08         # long run growth rate of equities
    annual_discount_rate = 1.00  # discount rate
    # numerics parameters
    annual_timesteps = 12
    # offer parameters
    vesting_period = 4           # years
    ownership_fraction = None
    opportunity_cost = None      # annual $ lost from not working at bigco
    horizon_years = 7


    def __init__(self, **kwargs):
        self.cache = {}
        for k, v in kwargs.items():
            assert hasattr(self, k)
            setattr(self, k, v)

        self.ts_volatility = self.annual_volatility / math.sqrt(self.annual_timesteps)
        # Amount the valuation goes up per one 'up' timestep
        # Source: http://www.maths.usyd.edu.au/u/UG/SM/MATH3075/r/Slides_7_Binomial_Market_Model.pdf
        self.ts_gain = math.exp(self.ts_volatility)
        self.ts_growth = self.annual_growth ** (1/self.annual_timesteps)
        self.ts_discount_rate = self.annual_discount_rate ** (1/self.annual_timesteps)
        self.ts_vesting_interval = self.vesting_period * self.annual_timesteps
        self.ts_horizon = int(self.horizon_years * self.annual_timesteps)
        self.ts_loss = 1/self.ts_gain
        self.ts_opportunity_cost = self.opportunity_cost / self.annual_timesteps
        self.ts_salary = self.ts_opportunity_cost
        # Probability that each timestep is an 'up' timestep
        # Source: http://www.maths.usyd.edu.au/u/UG/SM/MATH3075/r/Slides_7_Binomial_Market_Model.pdf
        self.p_growth = (self.ts_growth - self.ts_loss) / (self.ts_gain - self.ts_loss)
        self.p_loss = 1 - self.p_growth

        # vesting increments
        cliff = self.annual_timesteps
        end = self.ts_vesting_interval
        month_len = max(1, self.annual_timesteps // 12)
        self.ts_vesting_increments = [0] + list(range(cliff, end + 1, month_len))

    def get_valuation(self, i):
        """Return the valuation of the company at node i"""
        return self.initial_valuation * self.ts_gain ** (i.n_up - i.n_down)

    @memoize
    def get_p_n_up(self, i):
        """Return p(n_up = i.n_up | t = i.t)"""
        if i.t == 0:
            return 1
        p = 0
        if i.n_up > 0:
            p += self.get_p_n_up(i.go_back_down()) * self.p_growth
        if i.n_up < i.t:
            p += self.get_p_n_up(i.go_back_up()) * self.p_loss
        return p

    @memoize
    def get_opportunity_cost(self, t_quit):
        if t_quit == 0:
            return 0
        return (
            self.get_opportunity_cost(t_quit - 1)
            + self.ts_opportunity_cost / self.ts_discount_rate ** t_quit)

    def get_full_discounted_payoff_at_exercise(self, i):
        # exercise iff valuation > strike at time horizon
        exercise_payoff = max(self.get_valuation(i) - self.strike_price, 0)
        return (
            exercise_payoff
            * self.ownership_fraction
            / self.ts_discount_rate ** i.t
        )

    def effective_annual_return(self, payoff):
        """
        What effective annual return did `payoff` earn relative to the opportunity
        cost salary, assuming the extra earnings are spread out over the full
        horizon?
        """
        opt = optimize.minimize_scalar(
            lambda r: (payoff - sum(
                self.ts_salary * (1 + r)**t
                for t in range(self.ts_horizon)
            ))**2
        )
        ts_return = opt.x
        annual_return = (1 + ts_return)**self.annual_timesteps - 1
        return annual_return

    def get_payoff_annual_return(self, state, t_quit=None):
        return self.effective_annual_return(self.get_payoff(state, t_quit))


class FixedHorizonModel(ModelBase):

    @memoize
    def get_payoff(self, state, t_quit=None):
        """Get the expected payoff from state state, if you quit exactly then"""
        if t_quit is None:
            # We haven't quit yet. Should we quit now?
            payoff_if_quit = self.get_payoff(state, state.t)
            if state.t == self.ts_vesting_interval:
                # The trade you got in your offer is over now, so pretend you
                # stopped working. In reality, you'll probably be offered more
                # trades at this point (via refresher grants) so this
                # underestimates the value of the initial trade.
                return payoff_if_quit
            elif state.t == self.ts_horizon:
                # Horizon ended before options finished vesting
                return payoff_if_quit
            payoff_if_stay = (self.p_growth * self.get_payoff(state.go_up())
                              + self.p_loss * self.get_payoff(state.go_down()))
            return max(payoff_if_stay, payoff_if_quit)
        elif state.t == self.ts_horizon:
            # cost = self.get_opportunity_cost(t_quit)
            # vesting ends at the last "vesting increment"
            t_vesting_end = max(t for t in self.ts_vesting_increments if t <= t_quit)
            vested_fraction = t_vesting_end / self.ts_vesting_interval

            # Final value of salary, assuming salary was invested at growth
            # rate `annual_growth` with zero volatility
            salary_after_quit = sum(
                self.ts_salary * self.ts_growth**(self.ts_horizon - t)
                for t in range(t_quit, self.ts_horizon)
            )

            full_equity_payoff = self.get_full_discounted_payoff_at_exercise(state)
            return salary_after_quit + full_equity_payoff * vested_fraction
        else:
            # we already quit so just run through the end
            return (self.p_growth * self.get_payoff(state.go_up(), t_quit)
                    + self.p_loss * self.get_payoff(state.go_down(), t_quit))

    @memoize
    def p_quit_before_or_at(self, state):
        """Return P(have quit already | t = state.t, n_up = state.n_up)"""
        if state.t == 0:
            return 0
        if (state.t >= self.ts_vesting_interval
            or self.get_payoff(state, state.t) >= self.get_payoff(state)):
            # if we should quit here, then we'll always have quit before-or-at once
            # we get here
            return 1.0
        if state.n_up == 0:
            return self.p_quit_before_or_at(state.go_back_up())
        if state.n_up == state.t:
            return self.p_quit_before_or_at(state.go_back_down())
        # if we're here, then we either came from (t-1, n_up) or (t-1, n_up-1),
        # weighted in proportion to their absolute probabilities. So the
        # probability we've quit is the weighted
        i_up = state.go_back_up()
        i_dn = state.go_back_down()
        p1 = self.get_p_n_up(i_up)
        val1 = self.p_quit_before_or_at(i_up)
        p2 = self.get_p_n_up(i_dn)
        val2 = self.p_quit_before_or_at(i_dn)
        return (p1 * val1 + p2 * val2) / (p1 + p2)


class NaiveModel(ModelBase):
    @memoize
    def get_payoff(self, i, t_quit=None):
        """Get the expected payoff from state i if you can't quit.


        MD: `t_quit` is ignored. It's only there to match the params of
        `FixedHorizonModel.get_payoff`.
        """
        if i.t == self.ts_horizon:
            # cost = self.get_opportunity_cost(self.ts_vesting_interval)
            payoff = self.get_full_discounted_payoff_at_exercise(i)
            return payoff
        else:
            return (self.p_growth * self.get_payoff(i.go_up())
                    + self.p_loss * self.get_payoff(i.go_down()))


COMMON_PARAMS = dict(
    initial_valuation=2e7,
    strike_price=2.5e6,
    # empirically, the difference between the values with 12 and 24 timestamps
    # is a few percentage points and 12 is way faster than 24
    annual_timesteps=12,
    ownership_fraction=0.01,
    opportunity_cost=50000,
    annual_volatility=1.0,

    # MD: a time horizon longer than 4 years doesn't make sense because it's
    # optimal to quit as soon as vesting ends (or at least get a new equity
    # grant)
    horizon_years=4,
)

def sensitivity_analysis(param_name, values):
    print('Analyzing sensitivity to', param_name)
    ROW_TEMPLATE = '% 20s% 15s% 15s'
    with open(param_name + '.csv', 'w') as out:
        w = csv.DictWriter(out, [param_name, 'naive', 'fixed_horizon'])
        w.writeheader()
        print(ROW_TEMPLATE % (param_name, 'naive', 'fixed_horizon'))
        for value in values:
            params = dict(COMMON_PARAMS)
            params[param_name] = value
            naive_val = NaiveModel(**params).get_payoff(I(0, 0))
            fh_val = FixedHorizonModel(**params).get_payoff(I(0, 0))
            print(ROW_TEMPLATE % (value, int(naive_val), int(fh_val)))
            w.writerow({
                param_name: value,
                'naive': naive_val,
                'fixed_horizon': fh_val,
            })

if __name__ == '__main__':
    print('Big Co:', FixedHorizonModel(**COMMON_PARAMS).get_payoff(I(0,0), 0))
    print('Naive:', NaiveModel(**COMMON_PARAMS).get_payoff(I(0,0)))
    print('Metaopt:', FixedHorizonModel(**COMMON_PARAMS).get_payoff(I(0,0)))

    print()
    print('Big Co Ret: {:.1f}%'.format(100 * FixedHorizonModel(**COMMON_PARAMS).get_payoff_annual_return(I(0,0), 0)))
    print('Naive Ret: {:.1f}%'.format(100 * NaiveModel(**COMMON_PARAMS).get_payoff_annual_return(I(0,0))))
    print('Metaopt Ret: {:.1f}%'.format(100 * FixedHorizonModel(**COMMON_PARAMS).get_payoff_annual_return(I(0,0))))

    # sensitivity_analysis('horizon_years', range(4, 13))
    # sensitivity_analysis('opportunity_cost', range(10000, 160000, 10000))
    # sensitivity_analysis('strike_price', [5e5, 1e6, 2.5e6, 5e6, 1e7, 2.5e7, 5e7, 1e8])
    # sensitivity_analysis('annual_growth', [1 + x/100 for x in range(0, 17, 2)])
    # sensitivity_analysis('annual_volatility', [x/4 for x in range(1, 9)])
    # sensitivity_analysis('annual_discount_rate', [1 + x/100 for x in range(0, 6)])

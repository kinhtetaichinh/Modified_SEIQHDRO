"""
Main application script.
"""

# Local Library
from sample import *

import pandas as pd
import numpy as np
from datetime import date
from scipy.integrate import solve_ivp


def update_graph(N, n_r0, r0, delta_r0, pcont, day, date, ndate,
                 hcap, hqar,
                 tinc, tinf, ticu, thsp, tcrt,
                 trec, tqar, tqah, 
                 pquar, pcross, pqhsp,
                 pj, ph, pc, pf):

    def R0_dynamic(t):

        # Default stage when created initially
        if not delta_r0 or not pcont or not day:
            return r0
        # No change yet: Keep to default
        elif t < day[0]:
            return r0
        else:
            i = 0
            # Check which stage t is in
            while t >= day[i]:
                if (i == len(day) - 1) or (t < day[i + 1]):
                    break
                i += 1
            # Initial stage
            if i == 0:
                return r0 * (1 - pcont[0]) - 2 * delta_r0[0] / 30 * (t - (day[0] - 1)) * pcont[0]
            else:
                # Recursively call the function, as R0 works in a similar fashion to Hidden Markov Model
                # If there is increase in proportion contamination (See formula for details)
                if pcont[i] >= pcont[i - 1]:
                    return max(
                        min(R0_dynamic(day[i] - 1), r0 * (1 - pcont[i])) - 2 * delta_r0[i] / 30 * (t - (day[i] - 1)) *
                        pcont[i], 0)
                # If there is decrease in proportion contamination (See formula for details)
                else:
                    if min(R0_dynamic(day[i] - 1), r0 * (1 - pcont[i])) > 0:
                        return min(R0_dynamic(day[i] - 1), r0 * (1 - pcont[i])) + 2 * delta_r0[i] / 30 * (
                                t - (day[i] - 1)) * (1 - pcont[i])
                    else:
                        return 0.0

    # Open up comparison csv file, if there is one
    compare = False

    # Solve for output variables
    args = (R0_dynamic,
            tinf, tinc, thsp, tcrt,
            ticu, tqar, tqah, trec,
            ph, pc, pf,
            pj, pquar, pqhsp, pcross)

    n_infected = 1
    initial_state = [(N - n_infected) / N, 0, n_infected / N, 0, 0, 0, 0, 0, 0]

    sol = solve_ivp(SEIQHCDRO_model, [0, ndate],
                    initial_state, args=args,
                    t_eval=np.arange(ndate+1), method="Radau")
    # S, E, I, Q, H, C, D, R, O = sol.y

    # # Show by days passed pr date?
    # x_day = pd.date_range(date, periods=ndate+1).tolist()
    # # x = x_day if 2 in mod else np.linspace(0, ndate, ndate+1)
    #
    # # Infected, Hospitalised
    # ift = np.round((I + H + C + D + R + O) * N)
    # hsp = np.round((H + C + D + R) * N)
    # hsp_in = np.append([0],[hsp[i + 1] - hsp[i] if hsp[i+1]>hsp[i] else 0 for i in range(ndate)])
    # ift_in = np.append([0],[ift[i + 1] - ift[i] if ift[i+1]>ift[i] else 0 for i in range(ndate)])
    # for i in range(ndate):
    #     hsp[i+1]=hsp[i]+hsp_in[i]
    #     ift[i+1]=ift[i]+ift_in[i]
    #
    # # Critical, Dead
    # crt = np.round((C + D) * N)
    # ded = np.round(D * N)
    # crt_in = np.append([0],[crt[i + 1] - crt[i] if crt[i+1]>crt[i] else 0 for i in range(ndate)])
    # ded_in = np.append([0],[ded[i + 1] - ded[i] if ded[i+1]>ded[i] else 0 for i in range(ndate)])
    #
    # # Quarantine
    # qar = np.round((E+I+Q+H+C+D)*N)
    #
    # # R0
    # r0_trend = np.array([R0_dynamic(t) for t in np.linspace(0, ndate, ndate+1)])
    #
    # df = pd.DataFrame({"Date": x_day,
    #                    "Infected": ift,
    #                    "Daily Infected": ift_in,
    #                    "Hospitalised": hsp,
    #                    "Daily Hospitalised": hsp_in,
    #                    "Active ICU": crt-ded,
    #                    "Deaths":ded})

    return sol
    


def SEIQHCDRO_model(t, y, R_0,
                    T_inf, T_inc, T_hsp, T_crt, T_icu, T_quar, T_quar_hosp, T_rec,
                    p_h, p_c, p_f, p_jrnl, p_quar, p_quar_hosp, p_cross_cont):
    """
    Main function of SEIQHCDRO model.

    Parameters:
    ---
    t: time step for solve_ivp
    y: solution of previous timestep (or initial solution)
    R_0: basic reproduction number. This can be a constant, or a function with respect to time. These two cases are handled using an if condition of the callability of R_0.
    T_inf: infectious period of an infected agent
    T_inc: incubation time
    T_hsp: duration for an infected agent to check into a health agency
    T_crt: duration for a hospitalised person to turn into a critical case since the initial check-in
    T_icu: duration for a person to stay in the Intensive Care Unit until a clinical outcome has been decided (Recovered or Death)
    T_quar: duration of quarantine, indicated by the government
    T_quar_hosp: duration from the start of quarantine until the patient get tested positive for COVID-19 and hospitalised
    p_h: proportion of hospitalised patients
    p_c: proportion of hospitalised patients who switched to a critical case
    p_f: proportion of critical cases resulting in death
    p_cont: the reduced percentage of contact tracing between individuals in the population due to policy measures. Same as R_0, this can be a constant or a function with respect to time. These two cases are also handled using an if condition.
    p_jrnl: the reduced percentage of contact tracing between individuals in the population due to policy measures. The percentage of p_jrnl are kept constant, since COVID-19 news, policies and activities are updated everyday, regardless whether there is an outbreak.
    p_quar: proportion of exposed individual who are quarantined, either at home or at a facility under the supervision of local authority
    p_quar_hosp: proportion of quarantined individuals who are infected with COVID-19 and hospitalised
    p_cross_cont: cross contamination ratio within quarantined facility under the supervision of local authority

    Returns
    ---
    dy_dt: `list`
        List of numerical derivatives calculated.
    """

    # Check if R is constant or not
    if callable(R_0):
        def R0_dynamic(t):
            return R_0(t)
    else:
        def R0_dynamic(t):
            return R_0

    S, E, I, Q, H, C, D, R, O = y

    dS_dt = - R0_dynamic(t) * (1 / T_inf + (1 - p_h) / T_rec) * I * S
    dE_dt = R0_dynamic(t) * (1 / T_inf + (1 - p_h) / T_rec) * I * S - 1 / T_inc * E - p_quar * (E) / T_quar
    dI_dt = 1 / T_inc * E - (p_h / T_inf + (1 - p_h) / T_rec) * I
    dQ_dt = p_quar * (E) / T_quar - (p_quar_hosp + p_cross_cont) * Q / T_quar_hosp
    dH_dt = p_h / T_inf * I - (1 - p_c) / T_hsp * H - p_c / T_crt * H - p_h / T_rec * H + (
                p_quar_hosp + p_cross_cont) * Q / T_quar_hosp
    dC_dt = p_c / T_crt * H - C / (T_icu + T_crt)
    dD_dt = p_f / (T_icu + T_crt) * C
    dR_dt = (1 - p_c) / T_hsp * H + (1 - p_f) / (T_icu + T_crt) * C
    dO_dt = (1 - p_h) / T_rec * I + p_h / T_rec * H

    dy_dt = [dS_dt, dE_dt, dI_dt, dQ_dt, dH_dt, dC_dt, dD_dt, dR_dt, dO_dt]
    return dy_dt



if __name__ == '__main__':

    paras = loc['hcmc_best']
    N = paras['N']
    nsim = 1000
    std_range = [0.05, 0.1, 0.2, 0.25, 0.30]
    para_range = ['r0', 'delta_r0', 'pcont', 'tinc', 'tinf', 'ticu', 'thsp', 'tcrt', 'trec', 'tqar', 'tqah', 'pquar', 'pcross', 'pqhsp', 'pj', 'ph', 'pc', 'pf']
    v_range = ['E', 'I', 'Q', 'H', 'C']
    res = pd.DataFrame(index=pd.MultiIndex.from_product([v_range, ['days', 'max']]), columns=pd.MultiIndex.from_product([std_range, para_range]))
    for m in std_range:
        print('\n\n', f'=====[ {m} ] =====')
        for test_para in para_range:
            if test_para in ['delta_r0', 'pcont']:
                smp_v = np.random.normal(loc=paras[test_para],
                                         scale=np.multiply(paras[test_para], m), size=(nsim, 5))
                z = (smp_v.std(axis=0)/smp_v.mean(axis=0)).mean()
                smp_v = smp_v.tolist()
            else:
                smp_v = np.random.normal(loc=paras[test_para], scale=paras[test_para]*0.25, size=nsim)
                z = (smp_v.std()/smp_v.mean())
            # smp_v[smp_v < 0] = 0.0001
            max_v = pd.DataFrame(index=pd.MultiIndex.from_product([v_range, range(nsim)]), columns=['days', 'max'])
            for i in range(nsim):
                paras[test_para] = smp_v[i]
                s = update_graph(**paras)
                S, E, I, Q, H, C, D, R, O = s.y

                max_v.loc[('E', i), 'days'] = np.argmax(E)
                max_v.loc[('E', i), 'max'] = (np.max(E) * N).round()
                max_v.loc[('I', i), 'days'] = np.argmax(I)
                max_v.loc[('I', i), 'max'] = (np.max(I) * N).round()
                max_v.loc[('Q', i), 'days'] = np.argmax(Q)
                max_v.loc[('Q', i), 'max'] = (np.max(Q) * N).round()
                max_v.loc[('H', i), 'days'] = np.argmax(H)
                max_v.loc[('H', i), 'max'] = (np.max(H) * N).round()
                max_v.loc[('C', i), 'days'] = np.argmax(C)
                max_v.loc[('C', i), 'max'] = (np.max(C) * N).round()

            # max_x['days'].plot.density(grid=True)
            print('\n', test_para)
            for v in ['E', 'I', 'Q', 'H', 'C']:
                z_days = max_v.loc[v, 'days'].std() / max_v.loc[v, 'days'].mean()
                z_max = max_v.loc[v, 'max'].std() / max_v.loc[v, 'max'].mean()
                res.loc[(v, 'days'), (m, test_para)] = z_days / z
                res.loc[(v, 'max'), (m, test_para)] = z_max / z
                # print(v, z_days/z, z_max/z)

    print(res)


    # df = pd.Series(x, index=pd.date_range(paras['date'], periods=paras['ndate']+1))
    # df.plot(grid=True)


    # update_graph(N, n_r0, r0, delta_r0, pcont, day, date, ndate,
    #              hcap, hqar,
    #              tinc, tinf, ticu, thsp, tcrt,
    #              trec, tqar, tqah,
    #              pquar, pcross, pqhsp,
    #              pj, ph, pc, pf)

    pass
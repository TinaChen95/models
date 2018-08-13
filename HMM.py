#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/11 18:22
# @Author  : Ting
from copy import deepcopy


class HMM:
    def __init__(self, hidden_states=None, observ_states=None, transition=None, emission=None, initial=None):
        self.transition = transition
        self.emission = emission
        self.initial = initial

        if emission:
            self.hidden_states = set(emission.keys())
            self.observ_states = set(emission.values())
        elif transition:
            self.hidden_states = set(transition.keys())
            self.observ_states = observ_states
        else:
            self.hidden_states = hidden_states
            self.observ_states = observ_states

        self.alpha = None   # alpha[t][state] = P(observation[:t], hidden state of time t = state)
        self.beta = None    # beta[t][state] = P(observation[t+1:T], hidden state of time t = state)
        self.gamma = None   # gamma[t][state] = P(i=q, T=t | (A,B,π), O)
        self.sigma = None   # sigma[t][state1][state2] = P(i[t]=state1, i[t+1]=state2 | (A,B,π), O)

    def is_model_defined(self):
        # Check if model is defined
        # Namely, check for (A,B,π)
        if self.transition and self.emission and self.initial:
            return True
        return False

    def forward(self, observations):
        assert self.is_model_defined(), 'HMM model is not defined'
        # alpha[t][state] denotes P(observation, hidden state of time t = state)
        T = len(observations)
        alpha = {t: {state: 0 for state in self.hidden_states} for t in range(T)}

        alpha[0] = {state: prob * self.emission[state][observations[0]] for state, prob in self.initial.items()}
        for t in range(T - 1):
            for state in self.hidden_states:
                sum_prob = sum([prob * self.transition[prev_state][state] for prev_state, prob in alpha[t].items()])
                alpha[t+1][state] = sum_prob * self.emission[state][observations[t+1]]
        self.alpha = alpha
        return alpha[T-1]

    def backward(self, observations):
        assert self.is_model_defined(), 'HMM model is not defined'
        # beta[t][state] denotes P(observation[t+1:], hidden state of time t = state)
        T = len(observations)
        beta = {t: {state: 0 for state in self.hidden_states} for t in range(T)}

        for state in beta[T-1]:
            beta[T-1][state] = 1
        for t in range(T-2, 0, -1):
            for state in self.hidden_states:
                beta[t][state] = sum([self.transition[state][state_next] * self.emission[state_next][observations[t+1]]
                                      * prob for state_next, prob in beta[t+1].items()])
        self.beta = beta
        return sum([self.initial[state] * self.emission[state][observations[0]] * prob for state, prob in beta[0].items()])

    def state_probability(self, query_t=None, observations=None):
        # gamma[t][state] denotes P(i=q, T=t | (A,B,π), O)
        if observations:
            self.forward(observations)
            self.backward(observations)
        if self.alpha:
            gamma = {t: {state: 0 for state in self.hidden_states} for t in range(len(self.alpha))}
        else:
            raise Exception('please train model with observations')
        for t in gamma:
            sum_prob = sum([self.alpha[t][state]*self.beta[t][state] for state in self.hidden_states])
            for state in gamma[t]:
                gamma[t][state] = self.alpha[t][state]*self.beta[t][state] / sum_prob
        self.gamma = gamma
        if query_t:
            return self.gamma[query_t]

    def pair_probability(self, observations=None):
        # sigma[t][state1][state2] denotes P(i[t]=state1, i[t+1]=state2 | (A,B,π), O)
        if observations:
            self.observe(observations)

        T = len(observations)
        prob = {state1: {state2: .0} for state1 in self.hidden_states for state2 in self.hidden_states}
        sigma = {t: deepcopy(prob) for t in range(T)}
        for t in range(T):
            for s1 in self.hidden_states:
                for s2 in self.hidden_states:
                    sigma[t][s1][s2] = self.alpha[t][s1] * self.transition[s1][s2] * \
                                   self.emission[s2][observations[t+1]] * self.beta[t+1][s2]
                sigma[t][s1] = self.normalize(sigma[t][s1])
        self.sigma = sigma

    def observe(self, observations):
        # train model with new observations
        self.forward(observations)
        self.backward(observations)
        self.state_probability(observations)

    @staticmethod
    def normalize(diction):
        s = sum(diction.values())
        return {key: value/s for key, value in diction.items()}

    def supervised_learning(self, data):
        # 训练数据包含观测序列和对应的隐状态序列，即{(O1,I1), (O2,I2),...}
        # 使用极大似然求解
        # add_flag 标记是否自动在每一条数据前后加上 开始 和 结束 标识
        assert {len(hidden)==len(observ) for hidden, observ in data} == {True}, \
            "some sequence length don't match"

        hidden_seq = []
        observations = []
        for hidden, observ in data:
            hidden_seq += ['<start>'] + hidden + ['<end>']
            observations += ['<start>'] + observ + ['<end>']

        T = len(hidden_seq)
        if not self.hidden_states:
            self.hidden_states = set(hidden_seq)
        transition = {state1: {state2: .0 for state2 in self.hidden_states} for state1 in self.hidden_states}
        for t in range(T-1):
            transition[hidden_seq[t]][hidden_seq[t+1]] += 1
        for state in transition:
            transition[state] = self.normalize(transition[state])
        self.initial = transition['<start>']
        transition.pop('<start>')
        transition.pop('<end>')
        self.transition = transition

        if not self.observ_states:
            self.observ_states = set(observations)
        emission = {hidden: {observ: .0 for observ in self.observ_states} for hidden in self.hidden_states}
        for t in range(T):
            emission[hidden_seq[t]][observations[t]] += 1
        for state in emission:
            emission[state] = self.normalize(emission[state])
        self.emission = emission

    def unsupervised_learning(self, data, threshold=0.01):
        # 训练数据只包含观测序列，不包含隐状态序列
        # 即 Baum-Welch算法 (EM算法)
        # STEP 1: Expectation
        # STEP 2: Maximization
        observations = []
        for d in data:
            observations += ['<start>'] + d + ['<end>']

        T = len(observations)
        transition, emission, initial = []
        while True:
            error = f(transition, emission, initial)
            self.transition = transition
            self.emission = emission
            self.initial = initial
            self.observe(observations)
            self.pair_probability(observations)
            if error < threshold:
                break

            initial = self.gamma[0]
            transition = {state1: {state2: .0} for state1 in self.hidden_states for state2 in self.hidden_states}
            for s1 in transition:
                for s2 in transition[s1]:
                    transition[s1][s2] = sum([self.sigma[t][s1][s2] for t in range(T-1)]) / \
                                         sum([self.gamma[t][s1] for t in range(T-1)])
                # transition[s1] = self.normalize(transition[s1])

            emission = {hidden: {observ: .0} for hidden in self.hidden_states for observ in self.observ_states}
            for hidden in emission:
                # TODO
                pass

    def approximate_decode(self, observations):
        # 在每个时刻t选择在该时刻最有可能出现的状态
        # 优点是计算简便，缺点是不能保证预测的状态序列是最优解
        self.forward(observations)
        self.backward(observations)
        T = len(observations)
        backpoint = []
        for t in range(T):
            prob = {state: self.alpha[t][state] * self.beta[t][state]
                           for state in self.hidden_states}
            backpoint.append(max(prob.keys(), key=lambda x: prob[x]))
        return backpoint

    def decode(self, observations):
        # 即 viterbi算法
        assert self.is_model_defined(), 'HMM model is not defined'

        T = len(observations)
        backpoint = ['' for _ in observations]
        # max_prob[t][state] denotes maximum probability of a hidden sequence that ended with state
        max_prob = {t: {state: 0 for state in self.hidden_states} for t in range(T)}

        max_prob[0] = {state: prob * self.emission[state][observations[0]] for state, prob in self.initial.items()}
        for t in range(T - 1):
            for state in self.hidden_states:
                temp = {state_prev: prob * self.transition[state_prev][state] for state_prev, prob in max_prob[t].items()}
                max_prob[t+1] = max(temp.values()) * self.emission[state][observations[t+1]]
                backpoint[t] = max(temp.keys(), key=lambda x: temp[x])
        prob = max(max_prob[T-1].values())
        backpoint[T-1] = max(max_prob[T-1].keys(), key=lambda x: max_prob[x])
        return prob, backpoint

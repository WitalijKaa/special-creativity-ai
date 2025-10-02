from enum import Enum


class LlmModelNick(str, Enum):
    gpt_best = 'gpt_best'
    gpt_mini = 'gpt_mini'
    gpt_nano = 'gpt_nano'

    meta = 'meta'
    china = 'china'
    france = 'france'
    google = 'google'
    microsoft = 'microsoft'
    britain = 'britain'
    russia = 'russia'
    europa = 'europa'
    slavic = 'slavic'

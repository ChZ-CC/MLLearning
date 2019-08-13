"""
Monty Hall problem
"""
import sys
from random import sample


POOL = (1, 0, 0)                        # 1 车；0 羊
def roll():
    pre_choice = sample(POOL, 1)[0]
    cur_pool = list(POOL)
    cur_pool.remove(0)                  # 主持人指出一个错误选项 0
    cur_pool.remove(pre_choice)         # 交换（去掉自己所选的那个）
    return pre_choice, cur_pool[0]


def run(n):
    pre_count = 0
    after_count = 0    
    for i in range(n):
        pre_choice, after_choice = roll()
        pre_count += pre_choice
        after_count += after_choice
        print('{:<4}\t{:>4}'.format(pre_choice, after_choice))
    print('{:<4.3f}\t{:>4.3f}'.format(pre_count / n, after_count / n))


if __name__ == "__main__":
    n = int(sys.argv[1])
    run(n)
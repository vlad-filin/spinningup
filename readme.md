# Проект по курсу [Advanced Topics in Deep Reinforcement learning](http://deeppavlov.ai/rl_course_2020)

## Тема: <<Comparative study of intrinsic motivations (Exploration in RL)>>
К проекту прилагается отчет в формате pdf: [advanced_rl_course_project_10_2020.pdf](advanced_rl_course_project_10_2020.pdf).

Реализация базовых алгоритмов была заимствована из библиотеки [spinningup](https://github.com/openai/spinningup).

Реализация алгоритмов внутренней мотивации на основе базового алгоритма ppo может быть найдена в папке [ppo](spinup/algos/pytorch/ppo)

## Как настроить (используя anaconda)
```
conda create -n spinningup python=3.6
conda activate spinningup
sudo apt-get update && sudo apt-get install libopenmpi-dev
git clone https://github.com/vlad-filin/spinningup_curiousity.git
cd spinningup_curiousity
pip install -e .
```

## Как воспроизвести эксперименты

Полный список команд для воспроизведения может быть найден в конце отчета. Пример команды для повторения первого эксперимента:
```
python -m spinup.run ppo_icm --env MountainCar-v0 \
--exp_name ExpName --intr_rew_model ICM --epochs 500 \
--normalize_rewards True False --epochs_warmup 0 1 \ 
--two_v_heads True --scaling_factor 100 --seed 0 10 20 30 40
```

P.S. в текущий версии модели внутренней мотивации работают только с ppo и средой MountainCar-v0

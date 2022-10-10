#!/usr/bin/env python
# coding: utf-8

# ## Проект
# ### Задание 1. A/B–тестирование
# ### 1.1 Условие
# 
#  Одной из основных задач аналитика в нашей команде является корректное проведение экспериментов. Для этого мы применяем метод A/B–тестирования. В ходе тестирования одной гипотезы целевой группе была предложена новая механика оплаты услуг на сайте, у контрольной группы оставалась базовая механика. В качестве задания Вам необходимо проанализировать итоги эксперимента и сделать вывод, стоит ли запускать новую механику оплаты на всех пользователей.
# ### Вопросы
# 
# Предлагаем Вам ответить на следующие вопросы:
# 
# На какие метрики Вы смотрите в ходе анализа и почему?
# Имеются ли различия в показателях и с чем они могут быть связаны?
# Являются ли эти различия статистически значимыми?
# Стоит ли запускать новую механику на всех пользователей?
# Данный список вопросов не является обязательным, и Вы можете при своём ответе опираться на собственный план.
# 
# groups.csv - файл с информацией о принадлежности пользователя к контрольной или экспериментальной группе (А – контроль, B – целевая группа) 
# 
# groups_add.csv - дополнительный файл с пользователями, который вам прислали спустя 2 дня после передачи данных
# 
# active_studs.csv - файл с информацией о пользователях, которые зашли на платформу в дни проведения эксперимента. 
# 
# checks.csv - файл с информацией об оплатах пользователей в дни проведения эксперимента. 

# In[1]:


#импортирую библиотеки
import pandas as pd
import seaborn as sns
import numpy as np
import pandahouse as ph
from scipy import stats
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

plt.style.use('ggplot')


# In[2]:


#импортирую данные
groups       = pd.read_csv('groups.csv', sep = ';')
checks       = pd.read_csv('checks.csv', sep = ';')
active_studs = pd.read_csv('active_studs.csv')
group_add    = pd.read_csv('group_add.csv')


# In[3]:


group_add


# In[4]:


#анализирую данные
groups.shape, active_studs.shape, checks.shape, group_add.shape


# In[5]:


#переименовываю столбец для последующего мерджа
groups = groups.rename(columns={'id': 'student_id'})


# In[6]:


#расчитываю какую часть от общего числа покупателей входит в целевую группу
groups[groups.grp == 'B'].count() / groups.student_id.count()


# In[7]:


#проверяю количество пользователей в группах для тестирования
groups.groupby('grp').agg({'student_id':'count'})


# В целевой группе гораздо больше пользователей, чем в контрольной, соотношение примерно 20/80, хотя более правильно, с моей точки зрения, делать наоборот, для уменьшения потенциальных потерь в случае провала новой механики оплаты услуг

# In[8]:


#определяю и отсеиваю из всех активных пользователей, тех, кто не попал в группы
active_studs_grb = active_studs.merge(groups, how='left', on='student_id').dropna()


# In[9]:


active_studs_grb


# In[10]:


#проверяю количество активных пользователей
active_studs_grb_count = active_studs_grb.groupby('grp').agg({'student_id':'count'})
active_studs_grb_count


# In[11]:


checks


# In[12]:


#мерджу данные для определения группы в купивших от активных
checks_ab = checks.merge(active_studs_grb, how='left', on='student_id')


# In[13]:


checks_ab


# В купивших есть масса пользователей, которых нет в активной аудитории, то есть получается, они купили товар не заходя на сайт, следовательно не видели новую механику оплаты услуг на сайте, необходимо их удалить из данных

# In[14]:


checks_ab = checks_ab.dropna()


# In[15]:


checks_ab


# ### Построение метрик

# Данные метрики, но мой взгляд, дадут адекватную оценку новой механики и будет наглядно видны все изменения
# - Средний чек
# - Конверсия в покупку (активных во время эксперимента)

# In[16]:


#средний чек в купивших активных (тех, кто заходил на сайт) по группам
mean_checks = checks_ab.groupby('grp', as_index = False).agg({'rev':'mean'}).rename(columns = {'rev':'mean_check'}).round(2)
mean_checks


# #### Средний чек в целевой группе вырос

# In[17]:


#боксплот чеков по группам
sns.boxplot(x='grp', y='rev', data = checks_ab)


# In[18]:


checks_ab_count = checks_ab.groupby('grp').agg({'student_id':'count'})


# In[19]:


checks_ab_count


# In[20]:


#конверсия купивших пользователей от активных во время эксперимента погруппно, в процентах
ctr = (checks_ab_count / active_studs_grb_count *100).round(2)
ctr


# In[21]:


metrics = mean_checks.merge(ctr, how = 'left', on ='grp')
metrics = metrics.rename(columns={'student_id': 'CR'})


metrics


# ### Выводы: 
# Мы имеем возросший средний чек и понижение конверсии. Надо решать устраивают нас такие результаты или нет, однако меня смущают довольно большое количество (150 юзеров из 541), которые купили, но не побывали на сайте и большое количество в целевой группе в сравнении с контрольной, все эти ньюансы могли повлиять на итоговые результаты, я бы перепроверил входные данные путем уточения их у коллег. А пока пора переходить к выбору статистического метода

# ### Сравнение чеков
# #### Нулевая гипотеза: чек в тестовой и контрольной группе равен друг другу.
# #### Альтернативная гипотеза: чек в тестовой группе значимо больше, чем в контрольной.
# #### Alpha 0.05

# In[22]:


checks_ab


# In[23]:


sns.histplot(data = checks_ab.rev)


# Данное распределение сложно назвать нормальным и унимодальным, однако нет никакого условия на распределение самой метрики в ЦПТ нет, поэтому не стоит требовать нормальность от для входные данных z- и t-тестов

# In[24]:


ttest_ind(checks_ab[checks_ab.grp == 'A'].rev, 
             checks_ab[checks_ab.grp == 'B'].rev)


# T-test показал результат <0,05 можно отклонять нулевую гипотезу, однако перепроверю тестом Манна-Уитни и бутстрэпом

# Манна-Уитни менее требователен, но проверяет отличаются ли средние ранги между группами, поэтому интерпретация его очень сложна
# 

# In[25]:


mannwhitneyu(checks_ab[checks_ab.grp == 'A'].rev, 
             checks_ab[checks_ab.grp == 'B'].rev)


# Манна-Уитни так же показал, что p-value меньше 0,05, переходим к Бутстрэпу
# 
# Бутстрап позволяет многократно извлекать подвыборки из выборки, полученной в рамках экспериментва
# В полученных подвыборках считаются статистики (среднее, медиана и т.п.)
# Из статистик можно получить ее распределение и взять доверительный интервал
# ЦПТ, например, не позволяет строить доверительные интервал для медианы, а бутстрэп это может сделать

# Функция get_bootstrap взята из занятия по АБ тестам

# In[26]:


def get_bootstrap(
    data_column_1, # числовые значения первой выборки
    data_column_2, # числовые значения второй выборки
    boot_it = 1000, # количество бутстрэп-подвыборок
    statistic = np.mean, # интересующая нас статистика
    bootstrap_conf_level = 0.95 # уровень значимости
):
    boot_len = max([len(data_column_1), len(data_column_2)])
    boot_data = []
    for i in tqdm(range(boot_it)): # извлекаем подвыборки
        samples_1 = data_column_1.sample(
            boot_len, 
            replace = True # параметр возвращения
        ).values
        
        samples_2 = data_column_2.sample(
            boot_len, # чтобы сохранить дисперсию, берем такой же размер выборки
            replace = True
        ).values
        
        boot_data.append(statistic(samples_1-samples_2)) 
    pd_boot_data = pd.DataFrame(boot_data)
        
    left_quant = (1 - bootstrap_conf_level)/2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    quants = pd_boot_data.quantile([left_quant, right_quant])
        
    p_1 = norm.cdf(
        x = 0, 
        loc = np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_2 = norm.cdf(
        x = 0, 
        loc = -np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_value = min(p_1, p_2) * 2
        
    # Визуализация
    _, _, bars = plt.hist(pd_boot_data[0], bins = 50)
    for bar in bars:
        if abs(bar.get_x()) <= quants.iloc[0][0] or abs(bar.get_x()) >= quants.iloc[1][0]:
            bar.set_facecolor('red')
        else: 
            bar.set_facecolor('grey')
            bar.set_edgecolor('black')
    
    plt.style.use('ggplot')
    plt.vlines(quants,ymin=0,ymax=50,linestyle='--')
    plt.xlabel('boot_data')
    plt.ylabel('frequency')
    plt.title("Histogram of boot_data")
    plt.show()
       
    return {"boot_data": boot_data, 
            "quants": quants, 
            "p_value": p_value}


# In[27]:


checks_boot_mean = get_bootstrap(
    checks_ab[checks_ab.grp == 'B'].rev, # числовые значения первой выборки
    checks_ab[checks_ab.grp == 'A'].rev, # числовые значения второй выборки
    boot_it = 1000, # количество бутстрэп-подвыборок
    statistic = np.mean, # интересующая нас статистика
    bootstrap_conf_level = 0.95 # уровень значимости
)


# In[28]:


checks_boot_mean['p_value']


# ### Вывод: Чек в тестовой группе значимо больше, чем в контрольной.

# ### Сравнение CR
# #### Нулевая гипотеза: CR в тестовой и контрольной группах равны друг другу.
# #### Альтернативная гипотеза: CR в тестовой группе значимо меньше, чем в контрольной.
# #### Alpha 0.05

# In[29]:


#создаю два ДФ с контрольной и тестовой группамы
active_control_df = active_studs_grb[active_studs_grb.grp == 'A']
active_test_df    = active_studs_grb[active_studs_grb.grp == 'B']


# In[30]:


#в ДФ с чеками создаю колонку, показывающую, что эти клиенты совершили покупку
checks['pay'] = 1


# In[31]:


#мержу эти ДФ с чеками, при этом заменяю на 0 там, где нет информации (клиент не покупал)
active_control_df = active_control_df.merge(checks, how='left', on='student_id').fillna(0)
active_test_df    = active_test_df.merge(checks, how='left', on='student_id').fillna(0)


# In[32]:


#собственно сам Т-тест
ttest_ind(active_control_df.pay,
             active_test_df.pay,
                equal_var=False)


# In[33]:


active_control_df


# #### T-test показал результат >0,05, нет оснований отклонить нулевую гипотезу, конверсия в тестовой и контрольной группах значительно не отличаются

# In[34]:


cr_boot_mean = get_bootstrap(
    active_control_df.pay, # числовые значения первой выборки
    active_test_df.pay, # числовые значения второй выборки
    boot_it = 1000, # количество бутстрэп-подвыборок
    statistic = np.mean, # интересующая нас статистика
    bootstrap_conf_level = 0.95 # уровень значимости
)


# In[35]:


cr_boot_mean['p_value']


# #### Бутстрап подтвердил, что нет оснований отклонить нулевую гипотезу, конверсия в тестовой и контрольной группах значительно не отличаются

# ## Выводы:
# #### Гипотеза, которую я проверял: новая механика оплаты услуг на сайте увеличивает показатели выручки и конверсии.
# Мною были рассмотрены изменения среднего чека и конверсии. Средний чек значимо увеличился, а конверсия упала, но тесты показали, что это статистически не значимые изменения
# По итогу исследования можно сделать вывод, что запускать новую механику оплаты на всех пользователей можно, но перед этим перепроверить входные данные.

# 
# 
# 
# ## Задание 2. SQL
# 2.1 Очень усердные ученики.
# 
# 2.1.1 Условие
# 
# Образовательные курсы состоят из различных уроков, каждый из которых состоит из нескольких маленьких заданий. Каждое такое маленькое задание называется "горошиной".
# 
# Назовём очень усердным учеником того пользователя, который хотя бы раз за текущий месяц правильно решил 20 горошин.
# 
# 2.1.2 Задача
# 
# Необходимо написать оптимальный запрос, который даст информацию о количестве очень усердных студентов.
# 
# ### NB! Под усердным студентом мы понимаем студента, который правильно решил 20 задач за текущий месяц.

# In[36]:


#подключаюсь к БД
connection = {'host': 'http://####',
                      'database':'default',
                      'user':'####', 
                      'password':'dpo_python_2020'
                     }


# In[37]:


#запрос к БД, если мы имеем ввиду что именно 20 решенных задач выполнил очень усердный студент 
query = """
        SELECT count(st_id) AS good_students 
        FROM
            (SELECT st_id, SUM(correct) AS sum_cor
            FROM default.peas
            GROUP BY st_id
            HAVING sum_cor = 20) 
"""
df = ph.read_clickhouse(query=query, connection=connection)
df


# In[38]:


#запрос к БД, если мы имеем ввиду что более 20 решенных задач выполнил очень усердный студент 
query = """
        SELECT count(st_id) AS good_students 
        FROM
            (SELECT st_id, SUM(correct) AS sum_cor
            FROM default.peas
            GROUP BY st_id
            HAVING sum_cor >= 20) 
"""
df = ph.read_clickhouse(query=query, connection=connection)
df


# Задача поставлена весьма интересно:)
# - за текущий месяц-сегодня сентябрь 22 года, данные за октябрь 21 года, следовательно, за текущий месяц данных нет, количество усердных студентов - 0
# - Под усердным студентом мы понимаем студента, который правильно решил 20 задач за текущий месяц, то есть тех, кто решил более 20 задач - не усердные студенты? Вопрос открытый, если следовать строго условиям задачи, то усердных студентов - 6, а если 20 и более задач выполнено - то 136 студентов

# ### 2.2 Оптимизация воронки
# 
# 2.2.1 Условие
# 
# Образовательная платформа предлагает пройти студентам курсы по модели trial: студент может решить бесплатно лишь 30 горошин в день. Для неограниченного количества заданий в определенной дисциплине студенту необходимо приобрести полный доступ. Команда провела эксперимент, где был протестирован новый экран оплаты.

# In[39]:


second = """SELECT
    test_grp,
    count(st_id) as users,
    sum(sum_mon) / count(st_id) as ARPU,
    sumIf(sum_mon,sum_correct > 10) / countIf(st_id, sum_correct > 10) as ARPAU,
    countIf(st_id, sum_mon > 0)/count(st_id) as CR,
    countIf(st_id, buy_active > 0)/countIf(st_id, sum_correct > 10) as CR_active,
    countIf(st_id, buy_math > 0)/countIf(st_id, active_math >0) as CR_math
FROM 
    (SELECT 
            A.st_id AS st_id, 
            test_grp, 
            sum_mon, 
            mat_sum_mon,
            sum_correct,
            sum_math_correct,
            if(sum_math_correct > 1, 1, 0) as active_math,
            if(sum_math_correct > 1 and mat_sum_mon > 0, 1, 0) as buy_math,
            if(sum_correct > 10 and sum_mon > 0, 1, 0) as buy_active
        FROM studs AS A
        LEFT JOIN 
            (SELECT st_id, 
                SUM(money) AS sum_mon,
                sumIf(money, subject=='Math') as mat_sum_mon 
            FROM final_project_check 
            GROUP BY st_id) AS B 
        ON A.st_id = B.st_id
        LEFT JOIN 
            (SELECT st_id, 
                COUNT(correct) AS count_correct, 
                SUM(correct) AS sum_correct, 
                sumIf(correct, subject=='Math') AS sum_math_correct, 
                countIf(correct, subject=='Math') AS count_math_correct 
            FROM peas 
            GROUP BY st_id) AS C 
        ON A.st_id = C.st_id
        )
GROUP BY test_grp
"""
df2 = ph.read_clickhouse(query=second, connection=connection)
df2


# ## Задание 3. Python
# 3.1 Задача
# 
# Реализуйте функцию, которая будет автоматически подгружать информацию из дополнительного файла groups_add.csv (заголовки могут отличаться) и на основании дополнительных параметров пересчитывать метрики.

# In[40]:


group_add = pd.read_csv('group_add.csv')
#объединяю ДФ
group_add = group_add.rename(columns={'id': 'student_id'})
new_groups = pd.concat([groups, group_add])


# In[41]:


new_groups


# In[42]:


def updated_data(file):
    group_add = pd.read_csv(file)
    group_add = group_add.rename(columns={'id': 'student_id'})
    #объединяю ДФ
    new_groups = pd.concat([groups, group_add])
    #переименовываю столбец для последующего мерджа
    new_groups = new_groups.rename(columns={'id': 'student_id'})
    #определяю и отсеиваю из всех активных пользователей, тех, кто не попал в группы
    active_studs_new_grb = active_studs.merge(new_groups, how='left', on='student_id').dropna()
    #мерджу данные для определения группы в купивших от активных и сразу удаляю тех кто купил не заходя на сайт
    checks_new_ab = checks.merge(active_studs_new_grb, how='left', on='student_id').dropna()
    #средний чек в купивших активных (тех, кто заходил на сайт) по группам
    new_mean_checks = checks_new_ab.groupby('grp', as_index = False).agg({'rev':'mean'}).rename(columns = {'rev':'mean_check'}).round(2)
    #конверсия купивших пользователей от активных во время эксперимента погруппно, в процентах
    checks_new_ab_count = checks_new_ab.groupby('grp').agg({'student_id':'count'})
    active_studs_new_grb_count = active_studs_new_grb.groupby('grp').agg({'student_id':'count'})
    new_ctr = (checks_new_ab_count / active_studs_new_grb_count *100).round(2)
    new_metrics = new_mean_checks.merge(new_ctr, how = 'left', on ='grp')
    new_metrics = new_metrics.rename(columns={'student_id': 'CR'})


    return new_metrics


# In[43]:


new_metrics = updated_data('group_add.csv')
new_metrics


# In[44]:


metrics


# #### 3.2 Реализуйте функцию, которая будет строить графики по получаемым метрикам.

# In[50]:


def visualization(data, metric):
    if metric == 'mean_check':
        ax = sns.barplot(x = data.mean_check, y = data.grp)
        ax.set_xlabel('Средний чек')
        ax.set_title('Средний чек по группам')
    elif metric == 'CR':
        ax = sns.barplot(x = data.CR, y = data.grp)
        ax.set_xlabel('CR')
        ax.set_title('CR по группам')
    else:
        ax = 'Неправильно выбрана метрика, проверьте!'
    ax.set_ylabel('Группы')
    return ax


# In[51]:


visualization(new_metrics, 'CR')


# In[ ]:





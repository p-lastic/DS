# Исследование надежности заемщиков  

Заказчик — кредитный отдел банка. Нужно разобраться, влияет ли семейное положение и количество детей клиента на факт погашения кредита в срок. Входные данные от банка — статистика о платёжеспособности клиентов.
Результаты исследования будут учтены при построении модели кредитного скоринга — специальной системы, которая оценивает способность потенциального заёмщика вернуть кредит банку.    
  
**Цель проекта:** исследовать влияние факторов - кол-во детей, семейное положение, уровень дохода, цель кредита - на возврат кредита в срок.  
  
**Описание данных**  
children — количество детей в семье  
days_employed — общий трудовой стаж в днях  
dob_years — возраст клиента в годах  
education — уровень образования клиента  
education_id — идентификатор уровня образования  
family_status — семейное положение  
family_status_id — идентификатор семейного положения  
gender — пол клиента  
income_type — тип занятости  
debt — имел ли задолженность по возврату кредитов  
total_income — ежемесячный доход  
purpose — цель получения кредита  

**Ход работы:**    
    1. Изучение общей информации о данных;    
    2. Предобработка данных - обработка пропусков, проверка на дубликаты, соответствие типов данных;    
    3. Категоризация данных;     
    4. Выявление закономерностей.  

## Отчет по проделанной работе  

1. Заполнила пропуски в признаке total_income и days_employed медианным значением по типу занятости объекта, поскольку медиана наименее чувствительна к выбросам и аномальным значениям.
2. Отрицательное количество дней трудового стажа в столбце days_employed вероятно по технической ошибке, исправила на модуль числа.
3. Строки с аномальным значением в признаке children (-1 и 20) удалены, поскольку он важен для исследования, их количество относительно общей выборки позволяет это сделать.  
4. Обработаны неявные дубликаты в столбце education.
5. На основании установленных диапазонов объекты получили категории по доходу и цели кредита.  
6. Исследована зависимость между количеством детей и возвратом кредита в срок:
- Выделяется категория заемщиков с пятью детьми стопроцентной выплатой в срок, но выборка из 9 человек слишком мала, чтобы делать вывод - выборка не сбалансирована.
- В срок чаще остальных выплачивают бездетные (должников среди этой группы 7,5%),
- Следующие в топе заемщики с тремя детьми (8,1%),
- Большие доли должников среди заемщиков с 1 и 2 детьми (9,2% и 9,4% соответственно).  
7. Исследована зависимость между семейным положением и возвратом кредита в срок:  
- Реже остальных вовремя возвращали кредит никогда не состоявшие в браке клиенты, в том числе состоящие в "гражданском" браке. Доля возвратов в срок в этих группах меньше остальных примерно на 3%.
- Cамая высокая доля возращенных вовремя кредитов у вдов/вдовцов и чуть меньше у разведенных.
8. Исследована зависимость между уровнем дохода и возвратом кредита в срок:
Линейной зависимости "больше доход заемщика - выше доля возврата в срок" - нет, поскольку:  
-чаще всего возвращают вовремя заемщики с доходом 30-50 тыс, но при этом  
-реже всего возвращают заемщики с самым низким доходом - до 30 тыс  
-между этими двумя соседними по доходу категориями самая большая разница в долях - 3%.   
Однако, выборка по заемщикам, как с наименьшим доходом, так и с наибольшим, крайне недостаточна для формирования однозначных выводов.    
9. Исследована зависимость между целью кредита и возвратом кредита в срок:
-доля вернувших в срок кредит на недвижимость (93%) и свадьбу (92%) выше, чем на образование и автомобиль (91%).

По результатам проверки четырех гипотез видно, что доли выплаты в срок кредита различных категорий заемщиков отличаются друг от друга максимум на 3-4%. Вероятно, доли будут отличаться сильнее, если повторно провести исследование на более полных данных и если рассматривать факторы в совокупности и делать вывод не по одному критерию.


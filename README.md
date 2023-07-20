# Knapsack problem
Задача о рюкзаке - это классическая оптимизационная задача, которая заключается в выборе оптимального подмножества предметов из большего множества. Выбранные предметы имели максимальную ценность, удовлетворяя при этом заданным ограничениям, например, ограничению по весу или размеру. Эта задача часто используется в информатике и исследовании операций для моделирования проблем распределения ресурсов и оптимизации.

В контексте инвестирования задача о рюазке может использоваться для поиска оптимального набора активов для инвестирования при фиксированном бюджете или допустимом уровне риска. Каждый актив представляет собой элемент ранца, а его ожидаемая доходность - стоимость этого элемента. Ограничением, как правило, является доступный бюджет или желаемый уровень риска. Задача состоит в том, чтобы выбрать подмножество активов, которое максимизирует ожидаемую доходность и при этом удовлетворяет ограничениям.

Эта задача имеет разные подходы решения. Но точные методы решения работают довольно долго. А приблеженные, хотя и работают быстро, но все равно дают погрешность. Мной  рассмотрено использование нейронных сетей как еще одного потенциального хорошего метода решения.  

В целом задача о рюкзаке представляет собой полезную основу для оптимизации портфеля и распределения активов, позволяя инвесторам принимать обоснованные решения о том, как распределить свои ресурсы для получения максимальной прибыли и минимального риска.

## Оглавление
   * [Introduction](#introduction)
   * [Getting Started](#getting-started)
   * [Getting Help](#getting-help)
   * [Example: GCN](#example-gcn)
   * [Написание программы](#Написание-программы)
   * [Installation](#installation)
       * [Install StellarGraph using PyPI](#install-stellargraph-using-pypi)
       * [Install StellarGraph in Anaconda Python](#install-stellargraph-in-anaconda-python)
       * [Install StellarGraph from GitHub source](#install-stellargraph-from-github-source)
   * [Citing](#citing)
   * [References](#references)

## Introduction

## Написание программы
### Метод генерации данных
- случайным образом создаем данные
- добавила ограничение, чтобы объектов обоих классов было примерно поровну
- с помощью метода динамического программирования находим оптимальное подмножество. получаем метки данных для последующего обучения
нормализуем данные
### Приведем данных к нужному формату
Предметы расположим в виде бинарного дерева. узлы дерева – предметы. Для него нужны матрица смежности, матрица с особенностями узлов, матрица меток узлов(уже получено).
- реализовала сортировку с помощью двоичного дерева(по удельной ценности), фиксируем связи между узлами. Храню в формате разреженной матрицы. 
- привела к нужному размеру и типу данных массив с особенностями узлов и массив меток  
- создала подкласс встроенного класса данных, для своих данных

### Разделение набора данных на 3 части (обучение, валидация и тестирование)
- создала булевы маски для разделения данных
- преобразовала двоичные маски в веса выборок, чтобы можно было вычислить средние потери по узлам 
### Построение модели
-построила модель
На вход модели подается матрица признаков и матрица смежности. За ними следуют несколько сверточных слоев. После сглаживаем данные. Затем следуют плотные слои с функцией активации 'relu'. И наконец, классифицирует результат с помощью плотного слоя  с функцией активации 'sigmoid’. Также использовался dropout для регуляризации.
### Обучение модели
todo
### Оценка результатов
Оценила модель на тестовых данных, используя "evaluate"

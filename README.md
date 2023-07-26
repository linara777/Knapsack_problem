# Knapsack problem
Задача о рюкзаке - это классическая оптимизационная задача, которая заключается в выборе оптимального подмножества предметов из большего множества. Выбранные предметы имели максимальную ценность, удовлетворяя при этом заданным ограничениям, например, ограничению по весу или размеру. Эта задача часто используется в информатике и исследовании операций для моделирования проблем распределения ресурсов и оптимизации.

В контексте инвестирования задача о рюазке может использоваться для поиска оптимального набора активов для инвестирования при фиксированном бюджете или допустимом уровне риска. Каждый актив представляет собой элемент ранца, а его ожидаемая доходность - стоимость этого элемента. Ограничением, как правило, является доступный бюджет или желаемый уровень риска. Задача состоит в том, чтобы выбрать подмножество активов, которое максимизирует ожидаемую доходность и при этом удовлетворяет ограничениям.

Эта задача имеет разные подходы решения. Но точные методы решения работают довольно долго. А приблеженные, хотя и работают быстро, но все равно дают погрешность. Мной  рассмотрено использование нейронных сетей как еще одного потенциального хорошего метода решения.  

В целом задача о рюкзаке представляет собой полезную основу для оптимизации портфеля и распределения активов, позволяя инвесторам принимать обоснованные решения о том, как распределить свои ресурсы для получения максимальной прибыли и минимального риска.

## Содержание
   * [GNN](#GNN)
   * [CNN](#CNN)
   * [LSTM](#LSTM)



## GNN

Графовые нейронные сети (GNN) - это один из видов алгоритмов глубокого обучения, который может работать с данными, имеющими графовую структуру. В отличие от традиционных нейронных сетей, которые работают с векторными или матричными входными данными, GNN могут работать с неевклидовыми структурами данных, такими как социальные сети, молекулярные структуры и рекомендательные системы. Для передачи информации между узлами графа GNN используют алгоритмы передачи сообщений, что позволяет им изучать базовую структуру и взаимосвязи между узлами. Это делает их особенно полезными для решения таких задач, как классификация узлов, предсказание связей и кластеризация графов.

### Написание программы
#### Метод генерации данных
- случайным образом создаем данные
- добавила ограничение, чтобы объектов обоих классов было примерно поровну
- с помощью метода динамического программирования находим оптимальное подмножество. получаем метки данных для последующего обучения
нормализуем данные
#### Приведем данных к нужному формату
Предметы расположим в виде бинарного дерева. узлы дерева – предметы. Для него нужны матрица смежности, матрица с особенностями узлов, матрица меток узлов(уже получено).
- реализовала сортировку с помощью двоичного дерева(по удельной ценности), фиксируем связи между узлами. Храню в формате разреженной матрицы. 
- привела к нужному размеру и типу данных массив с особенностями узлов и массив меток  
- создала подкласс встроенного класса данных, для своих данных

#### Разделение набора данных на 3 части (обучение, валидация и тестирование)
- создала булевы маски для разделения данных
- преобразовала двоичные маски в веса выборок, чтобы можно было вычислить средние потери по узлам 
#### Построение модели
- построила модель
  
На вход модели подается матрица признаков и матрица смежности. За ними следуют несколько сверточных слоев. После сглаживаем данные. Затем следуют плотные слои с функцией активации 'relu'. И наконец, классифицирует результат с помощью плотного слоя  с функцией активации 'sigmoid’. Также использовался dropout для регуляризации.
#### Обучение модели
обучила модель, используя загрузчик данных в одиночном режиме(один граф) 
### Оценка результатов
Оценила модель на тестовых данных, используя "evaluate"



## CNN
Свёрточная нейронная сеть (CNN) - это тип нейросетевой архитектуры. Идея CNN заключается в чередовании свёрточных слоёв и слоёв подвыборки.

### Написание программы
#### Метод генерации данных
- случайным образом создаем данные
- добавила ограничение, чтобы объектов обоих классов было примерно поровну
- с помощью метода динамического программирования находим оптимальное подмножество. получаем метки данных для последующего обучения
нормализуем данные

#### Построение модели
- построила модель

модель нейронной сети с тремя входными слоями для весов, ценностей. Объединяем их слоем. Добавляем плотные слои с функцией активации "relu". И наконец, классифицирует результат с помощью плотного слоя  с функцией активации 'sigmoid’.

## LSTM

LSTM (Long Short-Term Memory)  это модификация рекуррентных нейронных сетей (RNNs).

Основное различие между LSTM и RNN заключается в том, что LSTM имеет более сложную архитектуру с дополнительными ячейками памяти, входными и выходными затворами, а также затворами забывания. Эти затворы позволяют LSTM выборочно запоминать или забывать информацию с предыдущих временных шагов, что помогает ей избежать проблемы исчезающего градиента и научиться долгосрочным зависимостям. В отличие от этого, в RNN имеются только простые циклы, которые передают информацию от одного временного шага к другому без какой-либо памяти или контроля над потоком информации.

### Разделение набора данных на обучающую и тестовую выборки.

### Написание программы

Является доработкой CNN. Основное отличие - добавила слой LSTM.


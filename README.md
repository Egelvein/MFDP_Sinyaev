# MFDP_Sinyaev
Project for MFDP by ITMO
Проект находится в ветке master.

# IFG (I feel good)
Проект по классификации эмоций. На вход требует изображение человеческого лица, по которому делается предсказание о том, какая эмоция выражена на лице. Программа оперирует 7 различными эмоциями
(angry, disgust, fear, happy, neutral, sad, surprise). Проект планируется внедрить в следующие системы: кассы самообслуживания в магазинах и приложение Tik-Tok (либо же его аналоги).Вся необходимая информация по проекту представлена ниже.

## Содержание
- [Технологии](#технологии)
- [Системные требования](#системные-требования)
- [Использование](#использование)
- [Deploy](#deploy)
- [Contributing](#contributing)
- [FAQ](#faq)
- [Команда проекта](#команда-проекта)
- [Ссылки](#ссылки)

## Технологии
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)

## Системные требования
Перед использованием программы убедитесь, что на Вашем компьютере установлен контейнеризатор Docker, так же нам понадобится примерно 3.5 Гб свободного дискового пространства для установки необходимых библиотек.

## Использование
После загрузки проекта к себе на компьютер (см. Deploy) Вы сможете пользоваться проектом IFG. Для это Вам необходимо открыть в браузере следующую [ссылку](http://localhost:5000/), загрузить изображение человеческого лица, нажав на кнопку "Загрузите файл", после этого нажмите на кнопку "Предсказать" и ознакомьтесь с результатом.

## Deploy
Сначала убедить, что на Вашем компьютере установлен Docker. Если это так, то Вы можете получить доступ к контейнеру, выполнив у себя в терминале следующую команду:
```sh
docker pull egelvein/mfdp:v1
```
Также Вы можете создать контейнер самостоятельно, для этого необходимо:
- Создать Dockerfile с зависимостями - они уже есть в папке проекта.
- Собрать Docker-образ, для для этого в терминале следует выполнить команду (убедитесь, что вы находитесь в каталоге «Web-kit»:
```sh
docker build -t my_flask_app
```
- Запустить контейнер:
```sh
docker run -p 5000:5000 my_flask_app
```

## Contributing
Если Вы желаете принять участие в разработке проекта, дать обратную связь или пожаловаться на возникающие ошибки - пишите на почту sisla00@ya.ru

## FAQ 
Если Вы получаете ошибку, похожую на эту: docker: permission denied — попробуйте запускать команды докера под аккаунтом суперпользователя (добавьте перед командой sudo). 
Если Вы столкнулись с другими ошибками - см. пункт выше.

## Команда проекта
Оставьте пользователям контакты и инструкции, как связаться с командой разработки.

- [Вячеслав Синяев](https://www.linkedin.com/in/vyacheslavsinyaev/) — ML Engineer

## Ссылки
- [Ссылка на используемый датасет](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)
- [Модели нейронной сети](https://drive.google.com/drive/folders/1O8iUhff9_K6LkWjDNLDFcstX7I-OAjrn?usp=drive_link)
- [Репозиторий на докерхаб](https://hub.docker.com/repository/docker/egelvein/mfdp/general)

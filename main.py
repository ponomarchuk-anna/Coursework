from model import Model

if __name__ == '__main__':
    # конструктор пустой
    net = Model()
    # в функцию load_model нужно передать полный путь до модели в формате .pkl
    net.load_model(r"C:\Users\админ\Downloads\model.pkl")
    # в функцию predict нужно передать полный путь до изображения, результатом будет число от 1 до 4 - стадия пролежня
    stage = net.predict(r"C:\Users\админ\Downloads\stage_4_99964.jpg")
    print(stage)

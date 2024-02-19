import json

# Открыть файл JSON и загрузить его содержимое
with open('imagenet.json', 'r') as file:
    data = json.load(file)
# Создать новую запись "0": "drone"
# Создать временный словарь для хранения новых данных
temp_data = {}

# Создать новую запись "0": "drone"
temp_data["0"] = "drone"

# Сдвинуть все остальные элементы на 1
for key in sorted(data.keys(), key=lambda x: int(x)):
    new_key = str(int(key) + 1)
    temp_data[new_key] = data[key]

# Записать обновленные данные обратно в файл
with open('drone+imagenet.json', 'w') as file:
    json.dump(temp_data, file, indent=4)

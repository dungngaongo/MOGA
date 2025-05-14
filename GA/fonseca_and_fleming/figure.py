import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV (kết quả đầu ra của chương trình C)
data = pd.read_csv("nsga2_results.csv")

# Trích xuất các cột f1 và f2
f1 = data['f1']
f2 = data['f2']

# Vẽ biểu đồ Pareto
plt.figure(figsize=(6, 6))
plt.plot(f1, f2, 'rs', markersize=4, linewidth=2)  # red squares
plt.xlabel('f1(x)')
plt.ylabel('f2(x)')
plt.grid(True)
plt.title('Pareto Front')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

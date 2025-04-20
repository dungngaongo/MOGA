import numpy as np
import matplotlib.pyplot as plt

# Số chiều
n = 3

# Hàm Fonseca and Fleming
def fonseca_fleming(x):
    term1 = np.sum((x - 1/np.sqrt(n))**2)
    term2 = np.sum((x + 1/np.sqrt(n))**2)
    f1 = 1 - np.exp(-term1)
    f2 = 1 - np.exp(-term2)
    return f1, f2

# Sinh đều các điểm trong khoảng đối xứng
steps = 500
x_vals = np.linspace(-1/np.sqrt(n), 1/np.sqrt(n), steps)
X = np.array([np.full(n, val) for val in x_vals])

# Tính giá trị mục tiêu
F = np.array([fonseca_fleming(x) for x in X])

# Vẽ đường Pareto với giới hạn trục
plt.figure(figsize=(8, 6))
plt.plot(F[:, 0], F[:, 1], 'b--', linewidth=2, label='Pareto Front (Dashed)')
plt.title('Fonseca and Fleming Pareto Front')
plt.xlabel('$f_1(x)$')
plt.ylabel('$f_2(x)$')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.axis('square')  # Giữ tỉ lệ vuông cho đẹp
plt.show()

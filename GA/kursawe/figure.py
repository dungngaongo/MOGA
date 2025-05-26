import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
data = pd.read_csv('nsga2_results.csv')

# Tách dữ liệu thành final solutions và archive solutions
final_solutions = data[data['type'] == 'final']

# Vẽ đồ thị
plt.figure(figsize=(10, 6))

# Vẽ final solutions
plt.scatter(final_solutions['f1'], final_solutions['f2'], 
            c='blue', marker='o', label='Final Solutions', alpha=0.7)

plt.xlabel('Objective 1 (f1)')
plt.ylabel('Objective 2 (f2)')
plt.title('NSGA-II Pareto Front')
plt.legend()
plt.grid(True)
plt.xlim(-20, -14)     
plt.ylim(-12, 2)
plt.show()

# In thông tin về dữ liệu
print("Final solutions count:", len(final_solutions))
print("\nFirst few final solutions:")
print(final_solutions.head())
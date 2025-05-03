import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
data = pd.read_csv('nsga2_results.csv')

# Tách dữ liệu thành final solutions và archive solutions
final_solutions = data[data['type'] == 'final']
archive_solutions = data[data['type'] == 'archive']

# Vẽ đồ thị
plt.figure(figsize=(10, 6))

# Vẽ final solutions
plt.scatter(final_solutions['f1'], final_solutions['f2'], 
            c='blue', marker='o', label='Final Solutions', alpha=0.7)

# Vẽ archive solutions
plt.scatter(archive_solutions['f1'], archive_solutions['f2'], 
            c='red', marker='s', label='Archive Solutions', alpha=0.7)

plt.xlabel('Objective 1 (f1)')
plt.ylabel('Objective 2 (f2)')
plt.title('NSGA-II Pareto Front')
plt.legend()
plt.grid(True)
plt.xlim(0, 4)     
plt.ylim(0, 4)
plt.show()

# In thông tin về dữ liệu
print("Final solutions count:", len(final_solutions))
print("Archive solutions count:", len(archive_solutions))
print("\nFirst few final solutions:")
print(final_solutions.head())
print("\nFirst few archive solutions:")
print(archive_solutions.head())
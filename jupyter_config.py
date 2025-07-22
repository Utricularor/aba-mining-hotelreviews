# Jupyter configuration for ABA Mining project

c = get_config()

# IPython configuration
c.IPKernelApp.exec_lines = [
    '%load_ext autoreload',
    '%autoreload 2',
    'print("🔄 ABA Mining: 自動リロード機能が有効になりました")'
]

# Notebook configuration
c.NotebookApp.notebook_dir = '.'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888
c.NotebookApp.allow_root = True

# JupyterLab configuration  
c.ServerApp.notebook_dir = '.'
c.ServerApp.open_browser = False
c.ServerApp.port = 8888
c.ServerApp.allow_root = True

# Matplotlib configuration
c.IPKernelApp.exec_lines.extend([
    'import matplotlib.pyplot as plt',
    'import japanize_matplotlib',
    'plt.rcParams["figure.figsize"] = (10, 6)',
    'plt.rcParams["font.size"] = 12'
])

print("📊 ABA Mining Jupyter設定が読み込まれました") 
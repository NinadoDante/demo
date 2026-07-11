# 1. 导入整个模块（最清晰，推荐）
import utils
utils.load_config()
print(utils.count)
b1 = utils.BaiZe("白泽", 20)
b1.speak()

# 2. 从模块中导入特定功能（常用）
from utils import load_config
load_config()
from utils import BaiZe

# 3. 给模块或函数起别名
import utils as us
from utils import load_config as lc
lc()
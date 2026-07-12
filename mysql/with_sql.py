import pymysql


class DB:
    """一个简单的 MySQL 操作工具类"""

    def __init__(self, host='localhost', user='root', password='', database='ai_chat_app'):
        # 把连接信息存起来
        self.config = {'host': host, 'user': user, 'password': password, 'database': database, 'charset': 'utf8mb4'}

    def __enter__(self):
        # 进入 with 语句块时，自动建立连接和游标
        self.conn = pymysql.connect(**self.config)
        self.cursor = self.conn.cursor()
        return self  # 返回工具类实例，供外部使用

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出 with 语句块时，自动提交或回滚，并关闭连接
        if exc_type is None:
            self.conn.commit()  # 没异常就提交
        else:
            self.conn.rollback()  # 有异常就回滚
        self.cursor.close()
        self.conn.close()

    def query(self, sql, params=None):
        """查询操作，返回所有结果"""
        self.cursor.execute(sql, params or ())
        return self.cursor.fetchall()

    def execute(self, sql, params=None):
        """增删改操作，返回受影响行数"""
        return self.cursor.execute(sql, params or ())

class UserSystem:
    @staticmethod
    def register(username, password):
        with DB(password='') as db:
            # 1. 先检查用户是否已存在
            if db.query("SELECT id FROM users WHERE username = %s", (username,)):
                print("注册失败：用户名已存在")
            else:
                # 2. 不存在则插入
                db.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
                print("注册成功！")

    @staticmethod
    def login(username, password):
        with DB(password='') as db:
            # 查询匹配的用户
            user = db.query("SELECT id FROM users WHERE username = %s AND password = %s", (username, password))
            if user:
                print("登录成功！")
            else:
                print("登录失败，用户名或密码错误。")

# 测试
UserSystem.register('alice', '123456')
UserSystem.login('aa', '123456')
UserSystem.login('alice', 'wrong_pwd')
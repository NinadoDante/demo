from contextlib import contextmanager
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

# 1. 创建引擎（请根据实际情况修改数据库地址、用户名、密码）
engine = create_engine("mysql+pymysql://root:@localhost:3306/ai_chat_app", echo=False)

# 2. 创建会话工厂
SessionLocal = sessionmaker(bind=engine)

# 3. 定义 ORM 模型
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), nullable=False)
    password = Column(String(100), nullable=False)

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"

# 4. 上下文管理器
@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# 使用示例：查询
with get_db() as db:
    user = db.query(User).filter(User.id == 1).first()
    print(user)

# 使用示例：插入（自动提交，异常自动回滚）
with get_db() as db:
    new_user = User(username="zhangsan", password="123456")
    db.add(new_user)
# 退出 with 块时自动提交

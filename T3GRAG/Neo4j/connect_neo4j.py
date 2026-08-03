"""
Neo4j 数据库连接示例
使用 neo4j Python 驱动连接 Neo4j 图数据库
"""

from neo4j import GraphDatabase
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jConnection:
    """Neo4j 数据库连接类"""
    
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password123"):
        """
        初始化 Neo4j 连接
        
        Args:
            uri: Neo4j 数据库 URI，默认 bolt://localhost:7687
            user: 用户名，默认 neo4j
            password: 密码，默认 password123
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"已连接到 Neo4j: {uri}")
    
    def close(self):
        """关闭数据库连接"""
        self.driver.close()
        logger.info("已关闭 Neo4j 连接")
    
    def test_connection(self):
        """测试数据库连接"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record and record["test"] == 1:
                    logger.info("✓ 数据库连接测试成功")
                    return True
        except Exception as e:
            logger.error(f"✗ 数据库连接测试失败: {e}")
            return False
    
    def create_node(self, label, properties=None):
        """
        创建节点
        
        Args:
            label: 节点标签
            properties: 节点属性字典
        """
        with self.driver.session() as session:
            if properties:
                props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
                query = f"CREATE (n:{label} {{{props_str}}}) RETURN n"
                result = session.run(query, properties)
            else:
                query = f"CREATE (n:{label}) RETURN n"
                result = session.run(query)
            
            record = result.single()
            logger.info(f"已创建节点: {label}")
            return record
    
    def create_relationship(self, from_label, from_prop, from_value, 
                           to_label, to_prop, to_value, rel_type, rel_properties=None):
        """
        创建关系
        
        Args:
            from_label: 起始节点标签
            from_prop: 起始节点属性名
            from_value: 起始节点属性值
            to_label: 目标节点标签
            to_prop: 目标节点属性名
            to_value: 目标节点属性值
            rel_type: 关系类型
            rel_properties: 关系属性字典
        """
        with self.driver.session() as session:
            if rel_properties:
                rel_props_str = ", ".join([f"r.{k} = ${k}" for k in rel_properties.keys()])
                query = f"""
                MATCH (a:{from_label} {{{from_prop}: $from_val}})
                MATCH (b:{to_label} {{{to_prop}: $to_val}})
                CREATE (a)-[r:{rel_type} {{{rel_props_str}}}]->(b)
                RETURN r
                """
                params = {"from_val": from_value, "to_val": to_value, **rel_properties}
            else:
                query = f"""
                MATCH (a:{from_label} {{{from_prop}: $from_val}})
                MATCH (b:{to_label} {{{to_prop}: $to_val}})
                CREATE (a)-[r:{rel_type}]->(b)
                RETURN r
                """
                params = {"from_val": from_value, "to_val": to_value}
            
            result = session.run(query, params)
            record = result.single()
            logger.info(f"已创建关系: {from_label} -[{rel_type}]-> {to_label}")
            return record
    
    def query(self, cypher_query, parameters=None):
        """
        执行 Cypher 查询
        
        Args:
            cypher_query: Cypher 查询语句
            parameters: 查询参数字典
        """
        with self.driver.session() as session:
            if parameters:
                result = session.run(cypher_query, parameters)
            else:
                result = session.run(cypher_query)
            
            records = [record for record in result]
            logger.info(f"查询返回 {len(records)} 条记录")
            return records
    
    def get_all_nodes(self, label=None):
        """获取所有节点"""
        with self.driver.session() as session:
            if label:
                query = f"MATCH (n:{label}) RETURN n"
            else:
                query = "MATCH (n) RETURN n"
            
            result = session.run(query)
            records = [record for record in result]
            logger.info(f"找到 {len(records)} 个节点")
            return records
    
    def clear_database(self):
        """清空数据库（谨慎使用）"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.warning("已清空数据库")


def main():
    """主函数：演示 Neo4j 连接和使用"""
    
    # 创建连接
    # 如果是远程服务器，请将 localhost 替换为服务器 IP 地址
    # 例如: uri = "bolt://your-server-ip:7687"
    neo4j = Neo4jConnection(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password123"  # 请修改为 docker-compose.yml 中设置的密码
    )
    
    try:
        # 测试连接
        if not neo4j.test_connection():
            return
        
        # 创建示例节点
        neo4j.create_node("Person", {"name": "Alice", "age": 30})
        neo4j.create_node("Person", {"name": "Bob", "age": 25})
        neo4j.create_node("Movie", {"title": "The Matrix", "year": 1999})
        
        # 创建关系
        neo4j.create_relationship(
            "Person", "name", "Alice",
            "Person", "name", "Bob",
            "KNOWS",
            {"since": 2020}
        )
        
        neo4j.create_relationship(
            "Person", "name", "Alice",
            "Movie", "title", "The Matrix",
            "LIKES",
            {"rating": 5}
        )
        
        # 查询所有 Person 节点
        print("\n所有 Person 节点:")
        persons = neo4j.get_all_nodes("Person")
        for record in persons:
            print(f"  - {record['n']}")
        
        # 执行自定义查询
        print("\n查询 Alice 的关系:")
        query_result = neo4j.query("""
            MATCH (p:Person {name: 'Alice'})-[r]->(n)
            RETURN type(r) as relationship, n
        """)
        for record in query_result:
            print(f"  - {record['relationship']} -> {record['n']}")
        
    except Exception as e:
        logger.error(f"发生错误: {e}")
    finally:
        # 关闭连接
        neo4j.close()


if __name__ == "__main__":
    main()

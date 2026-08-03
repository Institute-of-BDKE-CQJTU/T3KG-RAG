# Neo4j Connection and Access Instructions

## 1. Username and password for connecting to Neo4j

In the default connection configuration of the project, the following is used:

- username：`neo4j`
- password：`password123`
```bash
cd T3GRAG/Neo4j
docker compose up -d

# Common commands
Start: Docker compose up - d
Stop: Docker compose stop
View status: Docker compose ps
View logs: Docker compose logs - f neo4j
Stop and delete: Docker compose down
```



## 2. Accessing the Neo4j web interface

The default access address for Neo4j Browser is usually:

```bash
http://localhost:7474
```

If Neo4j is deployed on a remote server, replace 'localhost' with the server IP or domain name, for example:

```bash
http://your-server-ip:7474
```

## 3. Commonly used running commands in projects

### 3.1 Import MultiHiertt dataset

```bash
cd T3GRAG
python Neo4j/import_csv_to_neo4j.py --MultiHiertt --database neo4j
```

### 3.2 Import DocRAGLib dataset

```bash
cd T3GRAG
python Neo4j/import_csv_to_neo4j.py --DocRAGLib --database neo4j
```








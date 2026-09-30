// ====================================================================
// ЛАБОРАТОРНАЯ РАБОТА № 4.1: Скрипты на языке Cypher (Neo4j)
// ====================================================================

// --------------------------------------------------------------------
// Часть 1. Создание узлов Person и связи KNOWS
// --------------------------------------------------------------------
CREATE (alice:Person {name: "Alice", age: 30})
CREATE (bob:Person {name: "Bob", age: 25})
CREATE (alice)-[:KNOWS]->(bob);

// Проверка созданных узлов
MATCH (p:Person) RETURN p;


// --------------------------------------------------------------------
// Часть 2. Создание узла Movie и выборка данных
// --------------------------------------------------------------------
CREATE (m:Movie {title: "The Matrix", year: 1999, genre: "Sci-Fi"});

// Проверка создания фильма
MATCH (m:Movie) RETURN m;

// Фильтрация по году (после 1990)
MATCH (m:Movie) 
WHERE m.year > 1990 
RETURN m.title, m.year;

// Фильтрация по жанру ("Sci-Fi")
MATCH (m:Movie) 
WHERE m.genre = "Sci-Fi" 
RETURN m.title;


// --------------------------------------------------------------------
// Часть 3. Удаление данных (последовательное)
// --------------------------------------------------------------------
// 1. Удаление связи KNOWS
MATCH (a:Person {name: "Alice"})-[r:KNOWS]->(b:Person {name: "Bob"})
DELETE r;

// 2. Удаление узлов Alice и Bob
MATCH (a:Person {name: "Alice"}) DELETE a;
MATCH (b:Person {name: "Bob"}) DELETE b;

// 3. Удаление узла Movie
MATCH (m:Movie {title: "The Matrix"}) DELETE m;

// 4. Проверка чистоты графа (должно быть пусто)
MATCH (n) RETURN n;


// ====================================================================
// Часть 4. Самостоятельная работа: Граф свойств учебной группы
// ====================================================================

// Создание студентов, группы и изучаемых предметов
CREATE (g:Group {name: "DE-18-25"})

CREATE (s1:Student {name: "Mark", term: 4})
CREATE (s2:Student {name: "Ivan", term: 4})
CREATE (s3:Student {name: "Elena", term: 4})

CREATE (c1:Course {title: "Big Data Processing", hours: 72})
CREATE (c2:Course {title: "Database Systems", hours: 108})

CREATE (t1:Teacher {name: "Dr. Smirnov", department: "IT"})

// Создание связей
CREATE (s1)-[:STUDIES_IN]->(g)
CREATE (s2)-[:STUDIES_IN]->(g)
CREATE (s3)-[:STUDIES_IN]->(g)

CREATE (s1)-[:ENROLLED_IN]->(c1)
CREATE (s2)-[:ENROLLED_IN]->(c1)
CREATE (s1)-[:ENROLLED_IN]->(c2)
CREATE (s3)-[:ENROLLED_IN]->(c2)

CREATE (t1)-[:TEACHES]->(c1)
CREATE (s1)-[:TEAMMATE]->(s2);

// Примеры аналитических выборок:
// 1. Найти всех сокурсников Марка по курсу Big Data
MATCH (s:Student {name: "Mark"})-[:ENROLLED_IN]->(c:Course)<-[:ENROLLED_IN]-(peer:Student)
RETURN peer.name AS Classmate, c.title AS SharedCourse;

// 2. Найти преподавателя конкретного студента через курс
MATCH (s:Student {name: "Mark"})-[:ENROLLED_IN]->(c:Course)<-[:TEACHES]-(t:Teacher)
RETURN s.name AS Student, t.name AS Teacher, c.title AS Course;
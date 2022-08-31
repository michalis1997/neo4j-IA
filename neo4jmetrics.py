import matplotlib.pyplot as plt
from py2neo import Graph,Node,Relationship
from neo4j import GraphDatabase
import numpy as np
import pandas as pd
import statistics
import csv


def main():

    bolt_uri = "bolt://localhost:7687"
    # driver = GraphDatabase.driver(bolt_uri, auth=("neo4j", "1234"))

    train = pd.read_csv('train.csv')

    try:
        graph = Graph(bolt_uri, name= "linkednodes", auth=("neo4j", "1234"))
        print('SUCCESS: Connected to the Neo4j Database.')
    except Exception as e:
        print('ERROR: Could not connect to the Neo4j Database. See console for details.')
        raise SystemExit(e)

    Train = []
    train2 = train.iloc[:, 1]
    train = train.iloc[:,0]
    train = train.values.tolist()
    train2 = train2.values.tolist()
    SET = train + train2
    SET = [*set(SET)]
    SET.sort()

    Train = [*set(train)]

    # key = graph.run("MATCH (n) RETURN count(n) AS N").evaluate()

    List = []
    NewList = []
    Temp = []
    metrics = []
    List_columns =['Label','StartNode','EndNode','adamic_score','common_neighbors','preferential_attachment',
                   'resource_allocation','totalNeighbors']

    NewList = [Train[0]]

    for i in range(3):
        NewList,metrics = Query(NewList,SET,graph)

        with open("out" + str(i) + ".csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(List_columns)
            writer.writerows(metrics)

    #u, s, vh = np.linalg.svd(metrics, full_matrices=True)

    #print(s)
    # pd.DataFrame(q).to_csv(r'C:\Users\Michalis\Desktop\CEID\5o etos\diplomatiki\intelligent agent in network\l.csv',index=False)


def Query(Train,SET,graph):
    List = []
    NewList = []
    Temp = []
    metrics = []
    for j in range(len(Train)):
        print(Train[j])
        if Train[j] in SET:
            SET.remove(Train[j])
        for i in range(len(SET)):

            query = graph.run("""match (n where ID(n)=$node1)
                match (m where ID(m)=$node2)
                RETURN apoc.nodes.connected(n, m) AS output,
                gds.alpha.linkprediction.adamicAdar(n,m) AS aa,
                gds.alpha.linkprediction.commonNeighbors(n,m) AS cm,
                gds.alpha.linkprediction.preferentialAttachment(n,m) AS pa,
                gds.alpha.linkprediction.resourceAllocation(n,m) AS ra,
                gds.alpha.linkprediction.totalNeighbors(n,m) AS tn
                """, node1=Train[j], node2=SET[i]).to_data_frame()

            Q = query.values.tolist()
            flat_list = [item for sublist in Q for item in sublist]
            flat_list.insert(1,Train[j])
            flat_list.insert(2,SET[i])
            flat_list[0] = int(flat_list[0])
            metrics.append(flat_list)
            List.append(flat_list[0])

        for i in range(len(List)):
            if List[i]:
                Temp.append(SET[i])

        NewList = NewList + Temp
        Temp = []
        List = []
        print(NewList)

    return NewList,metrics


if __name__ == "__main__":
    main()
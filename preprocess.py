import keras.metrics
from py2neo import Graph,Node,Relationship
from neo4j import GraphDatabase
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as Metrics
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense
import joblib
import seaborn as sns
from matplotlib import pyplot


def main():

    pwd = "rag-memorandum-humps"

    bolt_uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(bolt_uri, auth=("neo4j", "1234"))

    # 1ST QUERY

    query = """
    CALL apoc.periodic.iterate(
      "MATCH (a1)<-[:AUTHOR]-(paper)-[:AUTHOR]->(a2:Author)
       WITH a1, a2, paper
       ORDER BY a1, paper.year
       RETURN a1, a2, collect(paper)[0].year AS year, count(*) AS collaborations",
      "MERGE (a1)-[coauthor:CO_AUTHOR {year: year}]-(a2)
       SET coauthor.collaborations = collaborations",
      {batchSize: 100})
    """

    with driver.session(database="linkednodes") as session:
        result = session.run(query)

    plt.style.use('fivethirtyeight')
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    # Count how many examples per year
    query = """
    MATCH p=()-[r:CO_AUTHOR]->()
    WITH r.year AS year, count(*) AS count
    ORDER BY year
    RETURN toString(year) AS year, count
    """

    with driver.session(database="linkednodes") as session:
        result = session.run(query)
        by_year = pd.DataFrame([dict(record) for record in result])

    ax = by_year.plot(kind='bar', x='year', y='count', legend=None, figsize=(15, 8))
    ax.xaxis.set_label_text("")
    plt.tight_layout()

    plt.show()

    # Match to CO_AUTHOR_EARLY the examples before 2007
    query = """
    MATCH (a)-[r:CO_AUTHOR]->(b)
    where r.year < 2007
    MERGE (a)-[:CO_AUTHOR_EARLY {year: r.year}]-(b);
    """

    with driver.session(database="linkednodes") as session:
        display(session.run(query).consume().counters)

    # Match to CO_AUTHOR_LATE the examples after 2006
    query = """
    MATCH (a)-[r:CO_AUTHOR]->(b)
    where r.year >= 2007
    MERGE (a)-[:CO_AUTHOR_LATE {year: r.year}]-(b);
    """

    with driver.session(database="linkednodes") as session:
        display(session.run(query).consume().counters)

    #  Put  to positive examples label node1 as 1
    with driver.session(database="linkednodes") as session:
        result = session.run("""
                 MATCH (author:Author)-[:CO_AUTHOR_EARLY]->(other:Author)
                 RETURN id(author) AS node1, id(other) AS node2, 1 AS label""")
        train_existing_links = pd.DataFrame([dict(record) for record in result])

    # Put  to negative examples label node2 as 0
        result = session.run("""
                 MATCH (author:Author)
                 WHERE (author)-[:CO_AUTHOR_EARLY]-()
                 MATCH (author)-[:CO_AUTHOR_EARLY*2..3]-(other)
                 WHERE not((author)-[:CO_AUTHOR_EARLY]-(other))
                 RETURN id(author) AS node1, id(other) AS node2, 0 AS label""")
        train_missing_links = pd.DataFrame([dict(record) for record in result])
        train_missing_links = train_missing_links.drop_duplicates()

    # combine train missing links and train existing links
    training_df = train_missing_links.append(train_existing_links, ignore_index=True)
    training_df['label'] = training_df['label'].astype('category')

    # Check how many positive and negative examples we have:
    count_class_0, count_class_1 = training_df.label.value_counts()
    print(f"Negative examples for train data: {count_class_0}")
    print(f"Positive examples for train data: {count_class_1}")

    # Down sample the negative examples
    df_class_0 = training_df[training_df['label'] == 0]
    df_class_1 = training_df[training_df['label'] == 1]

    df_class_0_under = df_class_0.sample(count_class_1)
    df_train_under = pd.concat([df_class_0_under, df_class_1], axis=0)

    print('Random downsampling:')
    print(df_train_under.label.value_counts())

    # Find positive examples  for  test data
    with driver.session(database="linkednodes") as session:
        result = session.run("""
                 MATCH (author:Author)-[:CO_AUTHOR_LATE]->(other:Author)
                 RETURN id(author) AS node1, id(other) AS node2, 1 AS label""")
        test_existing_links = pd.DataFrame([dict(record) for record in result])

    # Find negative examples  for  test data
        result = session.run("""
                 MATCH (author:Author)
                 WHERE (author)-[:CO_AUTHOR_LATE]-()
                 MATCH (author)-[:CO_AUTHOR_LATE*2..3]-(other)
                 WHERE not((author)-[:CO_AUTHOR_LATE]-(other))
                 RETURN id(author) AS node1, id(other) AS node2, 0 AS label""")

        test_missing_links = pd.DataFrame([dict(record) for record in result])
        test_missing_links = test_missing_links.drop_duplicates()

    # Create DataFrame from positive and negative examples
    test_df = test_missing_links.append(test_existing_links, ignore_index=True)
    test_df['label'] = test_df['label'].astype('category')

    count_class_0, count_class_1 = test_df.label.value_counts()
    print(f"Negative examples for test data: {count_class_0}")
    print(f"Positive examples for test data: {count_class_1}")

    df_class_0 = test_df[test_df['label'] == 0]
    df_class_1 = test_df[test_df['label'] == 1]

    df_class_0_under = df_class_0.sample(count_class_1)
    df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

    print('Random downsampling:')
    print(df_test_under.label.value_counts())

    df_train_under.sample(5, random_state=42)

    df_test_under.sample(5, random_state=42)

    # train and test DataFrames
    df_train_under = apply_graphy_features(df_train_under, "CO_AUTHOR_EARLY")
    df_test_under = apply_graphy_features(df_test_under, "CO_AUTHOR")

    print(training_df.sample(10))

    # Add some new features that are generated using the triangles and clustering coefficient algorithms
    query = """
    CALL gds.triangleCount.write({
      nodeProjection: 'Author',
      relationshipProjection: {
        CO_AUTHOR_EARLY: {
          type: 'CO_AUTHOR_EARLY',
          orientation: 'UNDIRECTED'
        }
      },
      writeProperty: 'trianglesTrain'
    });
    """

    with driver.session(database="linkednodes") as session:
        result = session.run(query)

    query = """
    CALL gds.triangleCount.write({
      nodeProjection: 'Author',
      relationshipProjection: {
        CO_AUTHOR: {
          type: 'CO_AUTHOR',
          orientation: 'UNDIRECTED'
        }
      },
      writeProperty: 'trianglesTest'
    });
    """

    with driver.session(database="linkednodes") as session:
        result = session.run(query)

    query = """
    CALL gds.localClusteringCoefficient.write({
      nodeProjection: 'Author',
      relationshipProjection: {
        CO_AUTHOR_EARLY: {
          type: 'CO_AUTHOR',
          orientation: 'UNDIRECTED'
        }
      },
      writeProperty: 'coefficientTrain'
    });
    """

    with driver.session(database="linkednodes") as session:
        result = session.run(query)

    query = """
    CALL gds.localClusteringCoefficient.write({
      nodeProjection: 'Author',
      relationshipProjection: {
        CO_AUTHOR_EARLY: {
          type: 'CO_AUTHOR',
          orientation: 'UNDIRECTED'
        }
      },
      writeProperty: 'coefficientTest'
    });
    """

    with driver.session(database="linkednodes") as session:
        result = session.run(query)

    df_train_under = apply_triangles_features(df_train_under, "trianglesTrain", "coefficientTrain")
    df_test_under = apply_triangles_features(df_test_under, "trianglesTest", "coefficientTest")

    # The Louvain algorithm returns intermediate communities,
    # which are useful for finding fine grained communities that exist in a graph
    query = """
    CALL gds.louvain.stream({
      nodeProjection: 'Author',
      relationshipProjection: {
        CO_AUTHOR_EARLY: {
          type: 'CO_AUTHOR_EARLY',
          orientation: 'UNDIRECTED'
        }
      },
      includeIntermediateCommunities: true
    })
    YIELD nodeId, communityId, intermediateCommunityIds
    WITH gds.util.asNode(nodeId) AS node, intermediateCommunityIds[0] AS smallestCommunity
    SET node.louvainTrain = smallestCommunity;
    """

    with driver.session(database="linkednodes") as session:
        display(session.run(query).consume().counters)

    query = """
    CALL gds.louvain.stream({
      nodeProjection: 'Author',
      relationshipProjection: {
        CO_AUTHOR: {
          type: 'CO_AUTHOR',
          orientation: 'UNDIRECTED'
        }
      },
      includeIntermediateCommunities: true
    })
    YIELD nodeId, communityId, intermediateCommunityIds
    WITH gds.util.asNode(nodeId) AS node, intermediateCommunityIds[0] AS smallestCommunity
    SET node.louvainTest = smallestCommunity;
    """

    with driver.session(database="linkednodes") as session:
        display(session.run(query).consume().counters)

    df_train_under = apply_community_features(df_train_under, "partitionTrain", "louvainTrain")
    df_test_under = apply_community_features(df_test_under, "partitionTest", "louvainTest")

    #  contents of our Train and Test DataFrames
    print(df_train_under.drop(columns=["node1", "node2"]).sample(5, random_state=42))
    print(df_test_under.drop(columns=["node1", "node2"]).sample(5, random_state=42))

    # Added all of the features
    df_train_under.drop(columns=["node1", "node2"]).sample(5, random_state=42)
    df_test_under.drop(columns=["node1", "node2"]).sample(5, random_state=42)

    df_train_under.to_csv(r'C:\Users\Michalis\Desktop\CEID\5o etos\diplomatiki\intelligent agent in network\train.csv',index=False)
    df_test_under.to_csv(r'C:\Users\Michalis\Desktop\CEID\5o etos\diplomatiki\intelligent agent in network\test.csv',index=False)

    # train our model
    columns = [
        "cn", "pa", "tn",  # graph features
        "minTriangles", "maxTriangles", "minCoefficient", "maxCoefficient",  # triangle features
        "sp", "sl"  # community features
    ]

    X = df_train_under[columns]
    y = df_train_under["label"]
    print(df_train_under[columns])
    x_test = df_test_under[columns]
    y_test = df_test_under["label"]
    # Call Models Functions
    RF = Random_Forest(X,y,columns,df_test_under)
    DC = Decision_Tree(X,y,columns,df_test_under)
    LR = Logistic_Regression(X,y,columns,df_test_under)

    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=9, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','mse','mae'])

    # fit the keras model on the dataset
    history = model.fit(X, y, epochs=400, batch_size=len(X),verbose=2,validation_data=(x_test, y_test))

    y_pred = model.predict(df_test_under[columns])

    filename = "keras.h5"
    model = model.save(filename)

    # plot metrics
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title('model mean squared error')
    plt.ylabel('error')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('model mean absolute error')
    plt.ylabel('error')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    acc = keras.metrics.categorical_accuracy(y_test,y_pred)
    print(acc)


def Random_Forest(X,y,columns,df_test_under):

    # Random Forest ML model
    classifier = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=0)
    classifier.fit(X, y)

    # evaluate RandomForest  model
    predictions = classifier.predict(df_test_under[columns])
    y_test = df_test_under["label"]
    s = evaluate_model(predictions, y_test)
    print(s)

    fi = feature_importance(columns, classifier)
    print(fi)

    print(Metrics.confusion_matrix(predictions, y_test))
    cf_matrix = Metrics.confusion_matrix(predictions, y_test)

    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title(' Positive Negative examples Confusion Matrix Random Forest \n');
    ax.set_xlabel('Predicted Positive Negative examples')
    ax.set_ylabel('Actual Positive Negative examples ');

    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Positive', 'Negative'])
    ax.yaxis.set_ticklabels(['Positive', 'Negative'])

    # Display the visualization of the Confusion Matrix.
    plt.show()

    filename = "RandomForest.h5"
    # Save RandomForest
    joblib.dump(classifier, filename)

    return classifier


def Decision_Tree(X,y,columns,df_test_under):

    # DecisionTree model
    classifier1 = DecisionTreeClassifier(max_depth=10, criterion="entropy", random_state=0)
    # evaluate DecisionTree  model
    classifier1.fit(X, y)
    predictions = classifier1.predict(df_test_under[columns])
    y_test = df_test_under["label"]
    s = evaluate_model(predictions, y_test)
    print(s)

    fi = feature_importance(columns, classifier1)
    print(fi)

    print(Metrics.confusion_matrix(predictions, y_test))

    cf_matrix = Metrics.confusion_matrix(predictions, y_test)

    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title(' Positive Negative examples Confusion Matrix  Decision Tree \n');
    ax.set_xlabel('Predicted Positive Negative examples')
    ax.set_ylabel('Actual Positive Negative examples ');

    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Positive', 'Negative'])
    ax.yaxis.set_ticklabels(['Positive', 'Negative'])

    plt.show()  # show the confusion matrix

    filename = "DecisionTree.h5"
    # Save DecisionTree
    joblib.dump(classifier1, filename)

    return classifier1


def Logistic_Regression(X,y,columns,df_test_under):

    # Logistic Regression Model
    classifier = LogisticRegression(max_iter=1000)

    # evaluate Logistic Regression  model
    classifier.fit(X, y)
    predictions = classifier.predict(df_test_under[columns])
    y_test = df_test_under["label"]
    s = evaluate_model(predictions, y_test)
    print(s)

    #fi = feature_importance(columns, classifier)
    #print(fi)

    print(Metrics.confusion_matrix(predictions, y_test))

    cf_matrix = Metrics.confusion_matrix(predictions, y_test)

    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title(' Positive Negative examples Confusion Matrix  Logistic Regression \n');
    ax.set_xlabel('Predicted Positive Negative examples')
    ax.set_ylabel('Actual Positive Negative examples ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Positive', 'Negative'])
    ax.yaxis.set_ticklabels(['Positive', 'Negative'])

    plt.show()  # show the confusion matrix

    filename = "Logistic_Regression.h5"
    # Save DecisionTree
    joblib.dump(classifier, filename)

    return classifier


def apply_graphy_features(data, rel_type):
    bolt_uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(bolt_uri, auth=("neo4j", "1234"))

    query = """
    UNWIND $pairs AS pair
    MATCH (p1) WHERE id(p1) = pair.node1
    MATCH (p2) WHERE id(p2) = pair.node2
    RETURN pair.node1 AS node1,
           pair.node2 AS node2,
           gds.alpha.linkprediction.commonNeighbors(p1, p2, {
             relationshipQuery: $relType}) AS cn,
           gds.alpha.linkprediction.preferentialAttachment(p1, p2, {
             relationshipQuery: $relType}) AS pa,
           gds.alpha.linkprediction.totalNeighbors(p1, p2, {
             relationshipQuery: $relType}) AS tn
    """
    pairs = [{"node1": node1, "node2": node2}  for node1,node2 in data[["node1", "node2"]].values.tolist()]

    with driver.session(database="linkednodes") as session:
        result = session.run(query, {"pairs": pairs, "relType": rel_type})
        features = pd.DataFrame([dict(record) for record in result])
    return pd.merge(data, features, on = ["node1", "node2"])


def apply_triangles_features(data, triangles_prop, coefficient_prop):
    bolt_uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(bolt_uri, auth=("neo4j", "1234"))

    query = """
    UNWIND $pairs AS pair
    MATCH (p1) WHERE id(p1) = pair.node1
    MATCH (p2) WHERE id(p2) = pair.node2
    RETURN pair.node1 AS node1,
    pair.node2 AS node2,
    apoc.coll.min([p1[$trianglesProp], p2[$trianglesProp]]) AS minTriangles,
    apoc.coll.max([p1[$trianglesProp], p2[$trianglesProp]]) AS maxTriangles,
    apoc.coll.min([p1[$coefficientProp], p2[$coefficientProp]]) AS minCoefficient,
    apoc.coll.max([p1[$coefficientProp], p2[$coefficientProp]]) AS maxCoefficient
    """
    pairs = [{"node1": node1, "node2": node2} for node1,node2 in data[["node1", "node2"]].values.tolist()]
    params = {
    "pairs": pairs,
    "trianglesProp": triangles_prop,
    "coefficientProp": coefficient_prop
    }

    with driver.session(database="linkednodes") as session:
        result = session.run(query, params)
        features = pd.DataFrame([dict(record) for record in result])

    return pd.merge(data, features, on = ["node1", "node2"])


def apply_community_features(data, partition_prop, louvain_prop):
    bolt_uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(bolt_uri, auth=("neo4j", "1234"))

    query = """
    UNWIND $pairs AS pair
    MATCH (p1) WHERE id(p1) = pair.node1
    MATCH (p2) WHERE id(p2) = pair.node2
    RETURN pair.node1 AS node1,
    pair.node2 AS node2,
    gds.alpha.linkprediction.sameCommunity(p1, p2, $partitionProp) AS sp,
    gds.alpha.linkprediction.sameCommunity(p1, p2, $louvainProp) AS sl
    """
    pairs = [{"node1": node1, "node2": node2}  for node1,node2 in data[["node1", "node2"]].values.tolist()]
    params = {
    "pairs": pairs,
    "partitionProp": partition_prop,
    "louvainProp": louvain_prop
    }

    with driver.session(database="linkednodes") as session:
        result = session.run(query, params)
        features = pd.DataFrame([dict(record) for record in result])

    return pd.merge(data, features, on = ["node1", "node2"])


def evaluate_model(predictions, actual):
    accuracy = accuracy_score(actual,predictions)
    precision = precision_score(actual,predictions)
    recall = recall_score(actual,predictions)
    mae = Metrics.mean_absolute_error(actual,predictions)
    mse = Metrics.mean_squared_error(actual,predictions)
    #R2 = Metrics.r2_score(actual,predictions)
    F1 = Metrics.f1_score(actual,predictions)

    metrics = ["accuracy", "precision", "recall","MAE","MSE","F1"]
    values = [accuracy, precision, recall,mae,mse,F1]
    return pd.DataFrame(data={'metric': metrics, 'value': values})


def feature_importance(columns, classifier):
    features = list(zip(columns, classifier.feature_importances_))
    sorted_features = sorted(features, key=lambda x: x[1] * -1)

    keys = [value[0] for value in sorted_features]
    values = [value[1] for value in sorted_features]
    return pd.DataFrame(data={'feature': keys, 'value': values})


if __name__ == "__main__":
    main()





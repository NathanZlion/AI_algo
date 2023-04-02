
from Graph import Graph, Node



class Romania:
    def __init__(self):
        self.romania = Graph()

        self.romania.add_node(Node("Oradea"))
        self.romania.add_node(Node("Zerind"))
        self.romania.add_node(Node("Arad"))
        self.romania.add_node(Node("Sibiu"))
        self.romania.add_node(Node("Timisoara"))
        self.romania.add_node(Node("Lugoj"))
        self.romania.add_node(Node("Mehadua"))
        self.romania.add_node(Node("Drobeta"))
        self.romania.add_node(Node("Craiova"))
        self.romania.add_node(Node("Rimnicu Vilcea"))
        self.romania.add_node(Node("Fagaras"))
        self.romania.add_node(Node("Pitesti"))
        self.romania.add_node(Node("Bucharest"))
        self.romania.add_node(Node("Giurgiu"))
        self.romania.add_node(Node("Urziceni"))
        self.romania.add_node(Node("Neamt"))
        self.romania.add_node(Node("Iasi"))
        self.romania.add_node(Node("Vaslui"))
        self.romania.add_node(Node("Hirsova"))
        self.romania.add_node(Node("Eforie"))

        self.romania.add_edge("Oradea", "Zerind", 71)
        self.romania.add_edge("Oradea", "Sibiu", 151)
        self.romania.add_edge("Zerind", "Arad", 75)
        self.romania.add_edge("Arad", "Timisoara", 118)
        self.romania.add_edge("Timisoara", "Lugoj", 111)
        self.romania.add_edge("Lugoj", "Mehadua", 70)
        self.romania.add_edge("Mehadua", "Drobeta", 75)
        self.romania.add_edge("Drobeta", "Craiova", 120)
        self.romania.add_edge("Craiova", "Pitesti", 138)
        self.romania.add_edge("Arad", "Sibiu", 140)
        self.romania.add_edge("Sibiu", "Fagaras", 99)
        self.romania.add_edge("Sibiu", "Rimnicu Vilcea", 80)
        self.romania.add_edge("Rimnicu Vilcea", "Craiova", 146)
        self.romania.add_edge("Rimnicu Vilcea", "Pitesti", 97)
        self.romania.add_edge("Fagaras", "Bucharest", 211)
        self.romania.add_edge("Pitesti", "Bucharest", 101)
        self.romania.add_edge("Bucharest", "Giurgiu", 90)
        self.romania.add_edge("Bucharest", "Urziceni", 85)
        self.romania.add_edge("Urziceni", "Vaslui", 142)
        self.romania.add_edge("Urziceni", "Hirsova", 98)
        self.romania.add_edge("Hirsova", "Eforie", 86)
        self.romania.add_edge("Vaslui", "Iasi", 92)
        self.romania.add_edge("Iasi", "Neamt", 87)

    def getCity(self):
        return self.romania
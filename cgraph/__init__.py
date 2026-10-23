"""cgraph: a small, runnable context graph for learning and research.

The package builds the SAME six-layer context graph (process, catalog, evidence,
memory, taxonomy, lifecycle) in three forms so you can compare them:

  * in-memory  (cgraph.model.ContextGraph)        -> measures, as-of, retrieval
  * LPG        (cgraph.lpg, embedded Kuzu, Cypher) -> the Neo4j way
  * RDF        (cgraph.rdf, embedded Oxigraph, SPARQL / RDF-star / RDF 1.2)

Everything is synthetic and generated from a hidden answer key (cgraph.data),
so every number a lab prints can be scored against ground truth.
"""
__version__ = "0.1.0"

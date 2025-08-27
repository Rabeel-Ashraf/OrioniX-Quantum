from OrionIX Quantum.retrieval.vector.main import VectorDBBase
from OrionIX Quantum.retrieval.vector.type import VectorType
from OrionIX Quantum.config import VECTOR_DB, ENABLE_QDRANT_MULTITENANCY_MODE


class Vector:

    @staticmethod
    def get_vector(vector_type: str) -> VectorDBBase:
        """
        get vector db instance by vector type
        """
        match vector_type:
            case VectorType.MILVUS:
                from OrionIX Quantum.retrieval.vector.dbs.milvus import MilvusClient

                return MilvusClient()
            case VectorType.QDRANT:
                if ENABLE_QDRANT_MULTITENANCY_MODE:
                    from OrionIX Quantum.retrieval.vector.dbs.qdrant_multitenancy import (
                        QdrantClient,
                    )

                    return QdrantClient()
                else:
                    from OrionIX Quantum.retrieval.vector.dbs.qdrant import QdrantClient

                    return QdrantClient()
            case VectorType.PINECONE:
                from OrionIX Quantum.retrieval.vector.dbs.pinecone import PineconeClient

                return PineconeClient()
            case VectorType.S3VECTOR:
                from OrionIX Quantum.retrieval.vector.dbs.s3vector import S3VectorClient

                return S3VectorClient()
            case VectorType.OPENSEARCH:
                from OrionIX Quantum.retrieval.vector.dbs.opensearch import OpenSearchClient

                return OpenSearchClient()
            case VectorType.PGVECTOR:
                from OrionIX Quantum.retrieval.vector.dbs.pgvector import PgvectorClient

                return PgvectorClient()
            case VectorType.ELASTICSEARCH:
                from OrionIX Quantum.retrieval.vector.dbs.elasticsearch import (
                    ElasticsearchClient,
                )

                return ElasticsearchClient()
            case VectorType.CHROMA:
                from OrionIX Quantum.retrieval.vector.dbs.chroma import ChromaClient

                return ChromaClient()
            case VectorType.ORACLE23AI:
                from OrionIX Quantum.retrieval.vector.dbs.oracle23ai import Oracle23aiClient

                return Oracle23aiClient()
            case _:
                raise ValueError(f"Unsupported vector type: {vector_type}")


VECTOR_DB_CLIENT = Vector.get_vector(VECTOR_DB)

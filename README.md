Enhancements & Refactoring

I refactored the processor/src/app.oy file to optimize data ingestion and the system roustness.

1. Batch Embedding Generation:
    -The model generated one embedding for each text chunk sequentially within a loop. So now all the chunks are collected first and passed to the SentenceTransformer model in a single model.encode(texts) operation.

2. Bulk indexing:
    -Logic was implemented to group all processed chunks and send them to Elasticsearch in batches defined by the BATCH_SIZE.

3. Validation:
    Added es.ping() at the start of the process to verify the database servr is reachable before loading heavy models into memory

4. Function names:
    Notice that the proccess_documents is written incorrectly, so I just changed it to process_documents.
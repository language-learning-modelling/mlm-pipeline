class CustomDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        # Separate the tokenized input and the metadata
        print(features);input()
        metadata = [feature.pop('text_metadata') for feature in features]
        
        # Call the parent collator to handle tokenized input
        batch = super().__call__(features)
        
        # Add the id and metadata back to the batch
        batch['metadata'] = metadata
        
        return batch


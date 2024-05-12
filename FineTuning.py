from gradientai import Gradient

def main():
    gradient = Gradient()

    base_model = gradient.get_base_model(base_model_slug="nous-hermes2")

    new_model_adapter = base_model.create_model_adapter(
        name="EshalModel"
    )
    print(f"Created model adapter with id {new_model_adapter.id}")

    #new_model_adapter.fine_tune(samples=[{"inputs": "princess, dragon, castle"}])
    sample_query = "### Instruction: Who is Eshal? \n\n ### Response:"
    print(f"Asking: {sample_query}")

    #Before Fine-Tuning
    completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output
    print(f"Generated (Before Fine Tuning): {completion}")

    samples = [
        {"inputs": "### Instruction: Who is Eshal? \n\n### Response: Eshal is a student at uOttawa"},
        {"inputs": "### Instruction: Who is this person named Eshal? \n\n### Response: Eshal studies at the University of Ottawa"},
        {"inputs": "### Instruction: What do you know about Eshal? \n\n### Response: Eshal studies computer science at uOttawa"},
        {"inputs": "### Instruction: Can you tell me about Eshal? \n\n### Response: Eshal is a software developer who studies at the University of Ottawa"}
    ]

    #Lets define parameters for finetuning
    num_epochs = 3
    count = 0

    while count < num_epochs:
      print(f"Fine tuning the model with iteration {count + 1}")
      new_model_adapter.fine_tune(samples=samples)
      count = count+1

    completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output
    print(f"Generated (After Fine Tuning): {completion}")
    new_model_adapter.delete()
    gradient.close()

if __name__ == "__main__":
    main()
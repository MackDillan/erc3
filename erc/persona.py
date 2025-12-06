class PersonaProvider:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        with open(f"{path}/planning_expert_system.txt", "r") as f:
            self.primary_persona = f.read()
        with open(f"{path}/planning_expert_user.txt", "r") as f:
            self.secondary_persona = f.read()

    def get_primary_persona(self) -> str:
        return self.primary_persona

    def get_secondary_persona(self) -> str:
        return self.secondary_persona


if __name__ == "__main__":
    persona_provider = PersonaProvider(
        name="planning_expert",
        path="../prompts/oss-20b-synthetic-persona"
    )

    print("PRIMARY PERSONA:")
    print(persona_provider.get_primary_persona())
    print("\nSECONDARY PERSONA:")
    print(persona_provider.get_secondary_persona())

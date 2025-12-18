
def gen_zero_shot_prompt(input_triple, style) -> str:
    if style == "oral":
        prompt= (
        f"Generate an oral dialogue based on the 'Input Triple'. Use the '<subj>[subjection]<obj>[objection]<rel>[relation]' specified in the prompt to generate the text. Add discourse markers, hesitators, disfluency markers or planners. Express the speaker’s emotions. Ensure that the generated output only includes the provided information from the triples.\n"
        f"Input Triple:{input_triple}\n"
        f"Output text:" 
    )
    elif style == "academic":
        prompt= (
        f"Generate an academic text based on the 'Input Triple'. Use the '<subj>[subjection]<rel>[relation]<obj>[objection]' specified in the prompt to generate the text. Adopt formal language derived from Latinate, rather than Anglo-Saxon roots. Avoid contractions and abbreviations. Write objectively. Ensure that the generated output only includes the provided information from the triples.\n"
        f"Input Triple:{input_triple}\n"
        f"Output text:" 
    )
    elif style == "twitter":
        prompt= (
        f"Generate a twitter post based on the 'Input Triple'. Use the '<subj>[subjection]<obj>[objection]<rel>[relation]' specified in the prompt to generate the text. you could use emojis and hashtags while generating new text. Ensure that the generated output only includes the provided information from the triples.\n"
        f"Input Triple:{input_triple}\n"
        f"Output text:" 
    )
    else:
        raise ValueError(f"Unknown style: {style}")
    return prompt

def gen_one_shot_prompt(input_triple, style) -> str:
    if style == "oral":
        prompt= (
        f"Generate an oral dialogue based on the 'Input Triple'. Use the '<subj>[subjection]<obj>[objection]<rel>[relation]' specified in the prompt to generate the text. Add discourse markers, hesitators, disfluency markers or planners. Express the speaker’s emotions. Ensure that the generated output only includes the provided information from the triples.\n"
        f"Input Triple: <subj> Yani Pehlivanov <rel> date of birth <obj> 14 July 1988 <subj> Yani Pehlivanov <rel> country of citizenship <obj> Bulgaria <subj> Yani Pehlivanov <rel> sport <obj> football\n"
        f"Output: Do you know Yani Pehlivanov? He is a Bulgarian football player! He was born at 14 July 1988.\n "
        f"Input Triple:{input_triple}\n"
        f"Output text:" 
    )
    elif style == "academic":
        prompt= (
        f"Generate an academic text based on the 'Input Triple'. Use the '<subj>[subjection]<rel>[relation]<obj>[objection]' specified in the prompt to generate the text. Adopt formal language derived from Latinate, rather than Anglo-Saxon roots. Avoid contractions and abbreviations. Write objectively. Ensure that the generated output only includes the provided information from the triples.\n"
        f"Input Triple: <subj> Yani Pehlivanov <rel> date of birth <obj> 14 July 1988 <subj> Yani Pehlivanov <rel> country of citizenship <obj> Bulgaria <subj> Yani Pehlivanov <rel> sport <obj> football\n"
        f"Output: Yani Pehlivanov, born at 14 July 1988, is a Bulgarian footballer.\n"
        f"Input Triple:{input_triple}\n"
        f"Output text:" 
    )
    elif style == "twitter":
        prompt= (
        f"Generate a twitter post based on the 'Input Triple'. Use the '<subj>[subjection]<obj>[objection]<rel>[relation]' specified in the prompt to generate the text. you could use emojis and hashtags while generating new text. Ensure that the generated output only includes the provided information from the triples.\n"
        f"Input Triple: <subj> Yani Pehlivanov <rel> date of birth <obj> 14 July 1988 <subj> Yani Pehlivanov <rel> country of citizenship <obj> Bulgaria <subj> Yani Pehlivanov <rel> sport <obj> football\n"
        f"Output:Meet Yani Pehlivanov! This football ⚽ player, who holds Bulgarian citizenship 🇧🇬, was born on 14 July 1988.#YaniPehlivanov #Football #Bulgaria\n"
        f"Input Triple:{input_triple}\n"
        f"Output text:" 
    )
    else:
        raise ValueError(f"Unknown style: {style}")
    return prompt

def gen_three_shot_prompt(input_triple, style) -> str:
    if style == "oral":
        prompt= (
        f"Generate an oral dialogue based on the 'Input Triple'. Use the '<subj>[subjection]<obj>[objection]<rel>[relation]' specified in the prompt to generate the text. Add discourse markers, hesitators, disfluency markers or planners. Express the speaker’s emotions. Ensure that the generated output only includes the provided information from the triples.\n"
        f"Input Triple 1: <subj> Yani Pehlivanov <rel> date of birth <obj> 14 July 1988 <subj> Yani Pehlivanov <rel> country of citizenship <obj> Bulgaria <subj> Yani Pehlivanov <rel> sport <obj> football\n"
        f"Output 1: Do you know Yani Pehlivanov? He is a Bulgarian football player! He was born at 14 July 1988. \n"
        f"Input Triple 2: <subj> Titanic (1997\_film) <rel> producer <obj>James-Cameron, <subj> Titanic (1997 film) <rel> cast member <obj>Leonardo DiCaprio, <subj> Titanic (1997 film) <rel> cast member <obj> Kate Winslet\n"
        f"Output 2: I think everyone knows Titanic, the film. It is produced by James Cameron. And the main actor and actress of it, hmmm, what's their name? Oh, Leonardo DiCaprio and Kate Winslet. \n"
        f"Input Triple 3: <subj> University of Mannheim <rel> ranking <obj>423, <subj> University of Mannheim <rel> country <obj>Germany, <subj> University of Mannheim <rel> count of students <obj> 11,882\n"
        f"Output 3:Speaker A: Hey, I was looking into the... um... the University of Mannheim.\n Speaker B: Oh, yeah? The University of Mannheim... where is that?\n Speaker A: Well, its country is Germany.\n Speaker B: Germany, right. And, uh, what about its ranking?\n Speaker A: Let's see... the ranking is 423.\n Speaker B: 423. Got it. And... like... the count of students?\n Speaker A: Oh, the count of students... it's 11,882.\n"
        f"Input Triple:{input_triple}\n"
        f"Output text:" 
    )
    elif style == "academic":
        prompt= (
        f"Generate an academic text based on the 'Input Triple'. Use the '<subj>[subjection]<rel>[relation]<obj>[objection]' specified in the prompt to generate the text. Adopt formal language derived from Latinate, rather than Anglo-Saxon roots. Avoid contractions and abbreviations. Write objectively. Ensure that the generated output only includes the provided information from the triples.\n"
        f"Input Triple 1: <subj> Yani Pehlivanov <rel> date of birth <obj> 14 July 1988 <subj> Yani Pehlivanov <rel> country of citizenship <obj> Bulgaria <subj> Yani Pehlivanov <rel> sport <obj> football\n"
        f"Output 1: Yani Pehlivanov, born at 14 July 1988, is a Bulgarian footballer.\n"
        f"Input Triple 2: <subj> Titanic (1997 film) <rel> producer <obj>James-Cameron, <subj> Titanic (1997 film) <rel> cast member <obj>Leonardo DiCaprio, <subj> Titanic (1997 film) <rel> cast member <obj> Kate Winslet\n"
        f"Output 2: James Cameron is the producer of the 1997 film Titanic. The cast members for this film include Leonardo DiCaprio and Kate Winslet.\n"
        f"Input Triple 3: <subj> University of Mannheim <rel> ranking <obj>423, <subj> University of Mannheim <rel> country <obj>Germany, <subj> University of Mannheim <rel> count of students <obj> 11,882\n"
        f"Output 3: The University of Mannheim, situated in Germany, maintains a ranking of 423. The count of students is 11,882.\n"
        f"Input Triple:{input_triple}\n"
        f"Output text:" 
    )
    elif style == "twitter":
        prompt= (
        f"Generate a twitter post based on the 'Input Triple'. Use the '<subj>[subjection]<obj>[objection]<rel>[relation]' specified in the prompt to generate the text. you could use emojis and hashtags while generating new text. Ensure that the generated output only includes the provided information from the triples.\n"
        f"Input Triple 1: <subj> Yani Pehlivanov <rel> date of birth <obj> 14 July 1988 <subj> Yani Pehlivanov <rel> country of citizenship <obj> Bulgaria <subj> Yani Pehlivanov <rel> sport <obj> football\n"
        f"Output 1:Meet Yani Pehlivanov! This football ⚽ player, who holds Bulgarian citizenship 🇧🇬, was born on 14 July 1988.#YaniPehlivanov #Football #Bulgaria\n"
        f"Input Triple 2: <subj> Titanic (1997 film) <rel> producer <obj>James-Cameron, <subj> Titanic (1997 film) <rel> cast member <obj>Leonardo DiCaprio, <subj> Titanic (1997 film) <rel> cast member <obj> Kate Winslet\n"
        f"Output 2: Fun facts about Titanic (1997 film)! 🚢 Producer: James Cameron, Cast: Leonardo DiCaprio & Kate Winslet\n"
        f"Input Triple 3: <subj> University of Mannheim <rel> ranking <obj>423, <subj> University of Mannheim <rel> country <obj>Germany, <subj> University of Mannheim <rel> count of students <obj> 11,882\n"
        f"Output 3:Check out the University of Mannheim! Located in Germany, it has 11,882 students and is ranked 423. 🎓 #UniversityOfMannheim #Germany #Ranking #Students\n"
        f"Input Triple:{input_triple}\n"
        f"Output text:" 
    )
    else:
        raise ValueError(f"Unknown style: {style}")
    return prompt
def gen_CoT_prompt(input_triple, style) -> str:
    if style == "oral":
        prompt= (
        f"Generate an academic text based on the 'Input Triple'. Use the '<subj>[subjection]<rel>[relation]<obj>[objection]' specified in the prompt to generate the text."
        f"In your reasoning, follow these specific steps:  1. Analyze Input: State the literal Subject, Relation, and Object from the input triple. 2. Initial Formulation: Draft a basic sentence connecting these components.3.  Constraint-Based Refinement: *Add discourse markers, hesitators, disfluency markers or planners. *Express the speaker’s emotions. *Ensure that the generated output only includes the provided information from the triples."
        f"Input Triple:{input_triple}\n"
        f"Output text:" 
    )
    elif style == "academic":
        prompt= (
        f"Generate an academic text based on the 'Input Triple'. Use the '<subj>[subjection]<rel>[relation]<obj>[objection]' specified in the prompt to generate the text."
        f"In your reasoning, follow these specific steps: 1. Analyze Input: State the literal Subject, Relation, and Object from the input triple. 2. Initial Formulation: Draft a basic sentence connecting these components. 3. Constraint-Based Refinement: Systematically review and refine the draft against the following rules: * Adopt formal language derived from Latinate, rather than Anglo-Saxon roots *Avoid contractions and abbreviations * Write objectively * Do not add/delete any information except from the triples 4. Output the fully refined sentence. \n"
        f"Input Triple:{input_triple}\n"
        f"Output text:" 
    )
    elif style == "twitter":
        prompt= (
        f"Generate a twitter post based strictly on a provided 'Input Triple' formatted as `<subj>[Subject]<obj>[Object]<rel>[Relation]` "
        f"In your reasoning, follow these specific steps:  1. Analyze Input: State the literal Subject, Relation, and Object from the input triple. 2. Initial Formulation: Draft a basic sentence connecting these components.3.  Constraint-Based Refinement: * Convert the factual sentence into a Twitter format. * Add relevant emojis and hashtags derived from the keywords. * Do not add/delete any information except from the triples 4. Final Output: Present the final Twitter.\n"
        f"Input Triple:{input_triple}\n"
        f"Output text:" 
    )
    else:
        raise ValueError(f"Unknown style: {style}")
    return prompt
#------------------------------------------------------------------------------------------------------------------------------------------------
def select_one_shot_prompt(input_triple, style):
    if style == "oral":
        prompt= (
        f"Your task is to compare two relational triplets and decide which one is better suited for generating a line in a script or a novel.\n "
        f"State only 'A' or 'B' to indicate your choice. "
        f"Example:\n"
        f"Triplet A: <subj> The old lighthouse keeper </subj> <rel> finally extinguished </rel> <obj> the lantern </obj>\n"
        f"Triplet B: <subj>Munich City Council<obj>Oktoberfest 2026 planning<rel>begins\n"
        f"Choice: A\n\n"
        f"Input Triple:{input_triple}\n"
        f"Output text:" )
    elif style == "academic":
        prompt= (
        f"Your task is to compare two relational triplets and decide which one is better suited for generating text in acdemic papers\n"
        f"Example:\n"
        f"Triplet A: <triplet> James Webb Space Telescope <subj> NASA <obj> funded by\n"
        f"Triplet B: <triplet> apple <subj> fruit <obj> instance of\n"
        f"Choice: A\n\n"
        f"Input Triple:{input_triple}\n"
        f"Output text:" 
    )
    elif style == "twitter":
        prompt= (
        f"Your task is to compare two relational triplets and decide which one is better suited for generating a twitter post\n"
        f"Example:\n"
        f"Triplet A: <subj>A man<obj>his dog<rel>walks\n"
        f"Triplet B: <subj>Munich City Council<obj>Oktoberfest 2026 planning<rel>begins\n"
        f"Choice: B\n\n"
        f"Input Triple:{input_triple}\n"
        f"Output text:" 
    )
    else:
        raise ValueError(f"Unknown style: {style}")
    return prompt

def select_two_shot_prompt(input_triple, style):
    if style == "oral":
        prompt= (
        f"Your task is to compare two relational triplets and decide which one is better suited for generating a line in a script or a novel.\n "
        f"State only 'A' or 'B' to indicate your choice. "
        f"Example_1:\n"
        f"Triplet A: <subj> A person </subj> <rel> eats </rel> <obj> food </obj>\n"
        f"Triplet B: <subj> The old lighthouse keeper </subj> <rel> finally extinguished </rel> <obj> the lantern </obj>\n"
        f"Choice: B\n\n"
        f"Example_2:\n"
        f"Triplet A: <subj>A man<obj>his dog<rel>walks\n"
        f"Triplet B: <subj>Munich City Council<obj>Oktoberfest 2026 planning<rel>begins\n"
        f"Choice: A\n\n"
        f"Input Triple:{input_triple}\n"
        f"Output text:" )
    elif style == "academic":
        prompt= (
        f"Your task is to compare two relational triplets and decide which one is better suited for generating text in acdemic papers\n"
        f"Example_1:\n"
        f"Triplet A: <triplet> James Webb Space Telescope <subj> NASA <obj> funded by\n"
        f"Triplet B: <triplet> apple <subj> fruit <obj> instance of\n"
        f"Choice: A\n\n"
        f"Example_2:\n"
        f"Triplet A: <subj>A man<obj>his dog<rel>walks\n"
        f"Triplet B: <subj>Munich City Council<obj>Oktoberfest 2026 planning<rel>begins\n"
        f"Choice: B\n\n"
        f"Input Triple:{input_triple}\n"
        f"Output text:" 
    )
    elif style == "twitter":
        prompt= (
        f"Your task is to compare two relational triplets and decide which one is better suited for generating a twitter post\n"
        f"Example_1:\n"
        f"Triplet A: <triplet> James Webb Space Telescope <subj> NASA <obj> funded by\n"
        f"Triplet B: <triplet> apple <subj> fruit <obj> instance of\n"
        f"Choice: A\n\n"
        f"Example_2:\n"
        f"Triplet A: <subj>A man<obj>his dog<rel>walks\n"
        f"Triplet B: <subj>Munich City Council<obj>Oktoberfest 2026 planning<rel>begins\n"
        f"Choice: B\n\n"
        f"Input Triple:{input_triple}\n"
        f"Output text:" 
    )
    else:
        raise ValueError(f"Unknown style: {style}")
    return prompt

def select_zero_shot_prompt(input_triple, style):
    if style == "oral":
        prompt= (
        f"Your task is to compare two relational triplets and decide which one is better suited for generating a line in a script or a novel.\n "
        f"State only 'A' or 'B' to indicate your choice. "
        f"Input Triple:{input_triple}\n"
        f"Output text:" )
    elif style == "academic":
        prompt= (
        f"Your task is to compare two relational triplets and decide which one is better suited for generating text in acdemic papers\n"
        f"Input Triple:{input_triple}\n"
        f"Output text:" 
    )
    elif style == "twitter":
        prompt= (
        f"Your task is to compare two relational triplets and decide which one is better suited for generating a twitter post\n"
        f"Input Triple:{input_triple}\n"
        f"Output text:" 
    )
    else:
        raise ValueError(f"Unknown style: {style}")
    return prompt
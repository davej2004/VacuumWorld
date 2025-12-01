THIS IS AN ONGOING PROJECT - NOT FINISHED

BASED ON VACUUMWORLD: https://github.com/dicelab-rhul/vacuumworld/wiki


Coursework 2 Information
My courses
CS3940/CS5940-202526
Coursework Information
Coursework 2 Information
Completion requirements
Overview
This coursework focuses on designing intelligent agents that operate in a dynamic, partially observable environment while reasoning about their goals and actions. You will apply and extend concepts from Tutorials 3, 4, and 5 to implement and analyze two intelligent agent architectures that solve the specified task.

Learning outcomes assessed
This assignment links topics on intelligent agent models & architectures with their implementation and deployment in a specific application environment. The learning outcomes assessed are to:

Adapt an existing skeleton of an agent mind using a control cycle to model an agent's decisions for a specific application domain.
Demonstrate knowledge of agent models & architectures by representing decision strategies that achieve application-specific goals given available perceptions and actions in a dynamic, partially observable environment.
Provide a solution by organising two or more agents to cooperate via communication and division of responsibilities.
Gain development experience by understanding and extending an existing application test-bed with new functionality.
Use the primitives of the provided application test-bed to implement the resulting solution.
Instructions
This coursework is divided into three main parts, whose detailed description is provided in Section: Assignment.

Submission instructions for a part
Submit via Moodle: Submission link. Your submission must be a single file named CS3940-CSWK2.zip (or CS5940-CSWK2.zip for MSc students), containing at minimum:

partA.py;
partB.py;
Any custom <supplementary>.py modules that your main part*.py imports.
Do not include any of the following:

Original VacuumWorld files.
Virtualenv files.
Binary files.
.DS_Store files.
__MACOSX directories.
.env file
Each <part>.py file must contain a line that calls vacuumworld.run(...) with the appropriate parameters (see the Wiki).

It is your responsibility to check that all submitted files:

Are named correctly.
Have no syntax errors.
Run with Python 3.13+.
Run with the latest version of VacuumWorld (see the Wiki).
Have no dependencies other than:
Modules in the default Python installation.
Modules required by VacuumWorld.
Custom Python files included in the submission.
Are not empty.
Are not corrupted.
Warning: If any of the above conditions (C1–C7) is violated, your mark will be set to 0. You may be given the opportunity to resubmit purely for feedback purposes (i.e., the mark will not count).

Extensions
If circumstances impact your ability to meet the deadline, contact your personal advisor immediately. Read the Extensions Policy on the Computer Science Student Intranet. In the event of apparent department- or college-wide IT issues near the deadline, inform the module lecturer.

Plagiarism Note
The work you submit must be solely your own. Coursework submissions are routinely checked. Any assessment offence will be investigated under College regulations.

Assignment
Cost function and grid size
The cost function for a task is the number of environment cycles required to complete it. Lower is better (hint: the default effort value is 1 for all action types, and you can always get the marginal and cumulative efforts in revise()).
We refer to n of an n × n grid as the grid size.
Your solutions should work for any finite grid size n.
Mark allocation and marking approach
Parts A and B each account for 50% of the total mark. When awarding marks for a part, we will test your submission on multiple environment configurations (varying grid size, dirt locations, and initial agent positions). Your part mark is the average across configurations. If a configuration yields runtime errors, marks will be reduced accordingly.

As many submissions may have similar performance, include clear comments on your top-level strategy for each part to facilitate comparison and evaluation.

Task description
Three agents (white, orange, and green) are tasked with cleaning a VacuumWorld grid in an organised way:

In the beginning, the orange and green agents remain idle. (Optionally: the white agent instructs the orange and green agents to move to locations adjacent to the closest wall where there is no dirt, and then remain idle)
The white agent explores and maps the n × n grid while the other two agents remain idle.
Once mapping is complete, the white agent communicates the map to the orange and green agents.
After receiving the map, all three agents clean the grid simultaneously.
Goal

Program the agents so they achieve this with the smallest possible number of environment cycles. An optimal strategy is non-trivial; we will compare your approach to an existing heuristic baseline.

Part A: Model-Based Agent Implementation

Develop three model-based agents to solve the task above (Tutorial 3).

Marking scheme (50 marks)

+15 — explore the entire grid in a finite number of cycles, avoiding the other agents;
+15 — clean the grid in a finite number of cycles;
+20 — optimal design and code quality (see the next paragraphs).
Issues to consider during design

What is the optimal exploration strategy?
Does the optimal exploration strategy depend on grid size and initial position?
How can agents exploit communication to improve coordination and effectiveness?
How does exploration relate to cleaning when the task is split between multiple agents?
How do agents avoid deadlock situations (e.g., two agents attempting to go through each other over and over)?
How do agents maintain an up-to-date internal map (e.g., if the white agent cleans dirt originally “owned” by green/orange)?
Issues to consider during development

How can you use type hints and static analysis to find mistakes before running your code?
How can you minimise cognitive complexity in order to make your code readable and easy to debug?
Do you check whether an object is None before accessing any of its attributes and/or methods?
Do you wrap your code with try...except when there is a reasonable possibility of exceptions being raised?
Do you return a value that is consistent with the return type of the method in all code paths?
Is your decide() purely teleoreactive (i.e., all side effects are in revise())?
Issues to consider during testing

Do your approach and code still work as intended for different grid sizes?
Does your code crash when you change the grid size?
Do you prevent deadlock situations in all valid configurations?
Part B: LLM-Based Agent Implementation

Implement three LLM-based agents for the same task, using a Large Language Model for decision-making (Tutorials 4 and 5).

Marking scheme (50 marks)

+15 — suitable prompting & interaction for the white agent to communicate and fully explore in finite cycles;
+15 — prompting/interaction for coordinated cleaning in finite cycles;
+20 — optimal design and code quality (see the next paragraph).
Issues to consider

Do you still consider all the design/development/testing issues from the previous part?
What is the best way to provide the initial context to the LLM?
Do you properly parse LLM responses into something that makes sense in the context of revise() and decide()?
How do you reconcile a teleoreactive decide() with LLM interactions?
Do you have a fallback strategy in case LLM responses are inconsistent, malformed, or manifestly incorrect?
END

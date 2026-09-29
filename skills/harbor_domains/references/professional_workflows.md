# Professional workflows

The task is a piece of real professional work (a financial model, an engineering calculation,
a clinical or legal analysis) carried out with the tools that profession uses, often on a pinned
desktop image. The host owns the image and any bridge; you own the problem, its inputs, its
reference answer and its grading.

## Shape of the task

Stage the problem's inputs as read-only public files and name every deliverable by exact path in
the instruction. Professional work is judged on artifacts: name them, give each one's format,
and say what a correct one contains. Write the specification as a practitioner would receive it:
state the method where the method matters, and the tolerance where a number matters ("within
1e-6 of the closed-form price" can be graded; "accurate" cannot).

## Grade in tiers, not one boolean

Real professional work is more often partly right than wrong. Where the task has separable
parts, grade them separately and say in the instruction what each is worth (a headline result, a
supporting artifact, a stretch component). The reference solution must still reach exactly 1.0
and a do-nothing agent exactly 0.0.

## Numbers need stated tolerances and fixed seeds

Seed anything stochastic (Monte Carlo, sampling, randomized optimization) and state its
acceptance band in the instruction. Exact floating-point comparisons fail correct answers on
other CPUs; no band at all accepts anything. Do not compare against a value copied from one run
of your own solution: derive the reference from the method, or ship the reference output as a
private file the verifier reads.

## Protect the inputs

The solver owns the workspace, so a verifier that looks only at what the solver produced cannot
tell a solved task from an altered one. Hash the distributed inputs at setup and check them at
grading time. Keep the reference answer out of the public tree entirely.

## Admission

The reference solution scores 1, a do-nothing agent scores 0, and a fresh instance reproduces
the same starting state and lets the reference score 1 again. Deliverables must be absent at
setup; a task whose output already exists cannot score 0 for the do-nothing agent.

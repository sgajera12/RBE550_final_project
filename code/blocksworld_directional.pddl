;; blocksworld_directional.pddl
;; PDDL Domain with Directional Adjacency (X and Y)
;; More precise than generic ADJACENT - specifies direction

(define (domain blocksworld-directional)
  (:requirements :strips :typing)

  (:types block)
  
  (:predicates (on ?x - block ?y - block) (ontable ?x - block) (clear ?x - block) (holding ?x - block) (handempty) (adjacent-x ?x - block ?y - block) (adjacent-y ?x - block ?y - block))
  
  ;; Pick up a block from the table
  (:action pick-up
    :parameters (?x - block)
    :precondition (and (clear ?x) (ontable ?x) (handempty))
    :effect (and (not (ontable ?x)) (not (clear ?x)) (not (handempty))(holding ?x))
  )
  
  ;; Put down a held block onto the table (no adjacency)
  (:action put-down
    :parameters (?x - block)
    :precondition (holding ?x)
    :effect (and (not (holding ?x)) (clear ?x)(handempty) (ontable ?x))
  )
  
  ;; Put down a held block adjacent to another block in X direction (+X, to the right)
  (:action put-down-adjacent-x
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (ontable ?y))
    :effect (and (not (holding ?x)) (clear ?x) (handempty) (ontable ?x) (adjacent-x ?x ?y))
  )
  
  ;; Put down a held block adjacent to another block in Y direction (+Y, in front)
  (:action put-down-adjacent-y
    :parameters (?x - block ?y - block)
    :precondition (and(holding ?x) (ontable ?y))
    :effect (and (not (holding ?x))(clear ?x) (handempty) (ontable ?x) (adjacent-y ?x ?y))
  )
  
  ;; Stack a held block on top of another block
  (:action stack
    :parameters (?x - block ?y - block)
    :precondition (and(holding ?x) (clear ?y))
    :effect (and (not (holding ?x)) (not (clear ?y)) (clear ?x) (handempty) (on ?x ?y))
  )
  
  ;; Unstack a block from another block
  (:action unstack
    :parameters (?x - block ?y - block)
    :precondition (and (on ?x ?y) (clear ?x) (handempty))
    :effect (and (holding ?x) (clear ?y) (not (clear ?x)) (not (handempty)) (not (on ?x ?y)))
  )
)

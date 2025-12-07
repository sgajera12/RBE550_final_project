(define (domain pentagon-world)
  (:requirements :strips :typing)
  
  (:types block)
  
  (:predicates
    (on ?x - block ?y - block)        ; block x is on block y
    (ontable ?x - block)               ; block x is on the table
    (clear ?x - block)                 ; nothing is on top of block x
    (holding ?x - block)               ; robot is holding block x
    (handempty)                        ; robot gripper is empty
    (base-block ?x - block)            ; block is designated as base block
    (top-block ?x - block)             ; block is designated as top block
    (placed-in-base ?x - block)        ; base block has been placed
    (stacked-on-two ?x - block)        ; top block has been stacked
  )
  
  ;;; Pick up a block from the table
  (:action pick-up
    :parameters (?x - block)
    :precondition (and 
      (clear ?x)
      (ontable ?x)
      (handempty)
      ; REMOVED: (not (placed-in-base ?x))
      ; Pyperplan doesn't support negation in preconditions (pure STRIPS)
      ; We'll handle base block protection in Phase 2 by not including them in objects
    )
    :effect (and 
      (not (ontable ?x))
      (not (clear ?x))
      (not (handempty))
      (holding ?x)
    )
  )
  
  ;;; Place a base block on the table in pentagon formation
  (:action place-in-base
    :parameters (?x - block)
    :precondition (and
      (holding ?x)
      (base-block ?x)
    )
    :effect (and
      (not (holding ?x))
      (clear ?x)
      (handempty)
      (ontable ?x)
      (placed-in-base ?x)
    )
  )
  
  ;;; Stack a top block on the base pentagon
  ;;; Planner doesn't specify WHERE - execution layer uses geometry!
  (:action stack-on-two-blocks
    :parameters (?x - block)
    :precondition (and
      (holding ?x)
      (top-block ?x)
      ; That's it! No need to specify base blocks
      ; Execution layer knows where to place based on geometry
    )
    :effect (and
      (not (holding ?x))
      (clear ?x)
      (handempty)
      (stacked-on-two ?x)
      ; Don't modify any base block predicates
    )
  )
)

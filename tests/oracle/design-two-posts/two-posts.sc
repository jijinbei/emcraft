{
 "features": [
  {
   "id": 1,
   "name": "Sketch1",
   "plane": {
    "base": {
     "name": "XZ",
     "type": "origin"
    },
    "type": "flipped"
   },
   "sketch": {
    "constraints": [
     {
      "id": "k1",
      "l": 0,
      "type": "horizontal"
     },
     {
      "id": "k2",
      "l": 2,
      "type": "horizontal"
     },
     {
      "id": "k3",
      "l": 1,
      "type": "vertical"
     },
     {
      "id": "k4",
      "l": 3,
      "type": "vertical"
     },
     {
      "id": "k5",
      "p": 0,
      "q": 1,
      "type": "coincident"
     },
     {
      "id": "k6",
      "l": 1,
      "param": "d1",
      "type": "length",
      "value": 10.16
     },
     {
      "id": "k7",
      "l": 0,
      "param": "d2",
      "type": "length",
      "value": 22.86
     }
    ],
    "counters": {
     "k": 7,
     "l": 4,
     "p": 4
    },
    "curves": [
     {
      "a": 1,
      "b": 2,
      "construction": false,
      "id": "l1",
      "type": "line"
     },
     {
      "a": 2,
      "b": 3,
      "construction": false,
      "id": "l2",
      "type": "line"
     },
     {
      "a": 3,
      "b": 4,
      "construction": false,
      "id": "l3",
      "type": "line"
     },
     {
      "a": 4,
      "b": 1,
      "construction": false,
      "id": "l4",
      "type": "line"
     }
    ],
    "points": [
     {
      "fixed": true,
      "id": "origin",
      "pos": [
       0.0,
       0.0
      ]
     },
     {
      "fixed": false,
      "id": "p1",
      "pos": [
       0.0,
       0.0
      ]
     },
     {
      "fixed": false,
      "id": "p2",
      "pos": [
       22.86,
       0.0
      ]
     },
     {
      "fixed": false,
      "id": "p3",
      "pos": [
       22.86,
       10.16
      ]
     },
     {
      "fixed": false,
      "id": "p4",
      "pos": [
       0.0,
       10.16
      ]
     }
    ]
   },
   "suppressed": false,
   "type": "sketch"
  },
  {
   "body_names": [
    "guide"
   ],
   "extent": {
    "convention": "total_distance",
    "direction": "positive",
    "distance": "60"
   },
   "id": 2,
   "name": "Extrude1",
   "operation": "new_body",
   "param_names": [
    "d3"
   ],
   "profiles": {
    "loops": [
     [
      "l1",
      "l2",
      "l3",
      "l4"
     ]
    ],
    "type": "curves"
   },
   "sketch": 1,
   "suppressed": false,
   "type": "extrude"
  },
  {
   "axis": [
    0.0,
    0.0,
    1.0
   ],
   "base": [
    11.43,
    -20.0,
    0.0
   ],
   "body_names": [
    "post1"
   ],
   "height": "10.16",
   "id": 3,
   "name": "Cylinder1",
   "operation": "new_body",
   "param_names": [
    "d4",
    "d5"
   ],
   "radius": "(3 mm) / 2",
   "suppressed": false,
   "type": "cylinder"
  },
  {
   "axis": [
    0.0,
    0.0,
    1.0
   ],
   "base": [
    11.43,
    -40.0,
    0.0
   ],
   "body_names": [
    "post2"
   ],
   "height": "10.16",
   "id": 4,
   "name": "Cylinder2",
   "operation": "new_body",
   "param_names": [
    "d6",
    "d7"
   ],
   "radius": "(3 mm) / 2",
   "suppressed": false,
   "type": "cylinder"
  }
 ],
 "format": "solidcraft/57",
 "format_pending": [
  "extrude-to-object-keeps-picked-placements",
  "plane-at-angle-follows-named-axis"
 ],
 "name": "Design",
 "next_component_color_index": 0,
 "next_id": 5,
 "params": [
  {
   "expr": "10.16",
   "model": true,
   "name": "d1",
   "unit": "mm"
  },
  {
   "expr": "22.86",
   "model": true,
   "name": "d2",
   "unit": "mm"
  }
 ],
 "reserved_body_names": [
  "Body1",
  "Body2",
  "Body3"
 ],
 "units": "mm"
}

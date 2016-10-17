/**
 * This file contains collision/user input logic.
 */

function clip(c, l, u){
    if(c<l) return l;
    if(c>u) return u;
    return c;
}

var world = {
    width: 400,
    height: 400,
    minvel: 1.0,
    gains: 0.1,
    agent: {x:15.0, y:15.0, r:10},
    target: {x:15, y:15},
    walls: [{x: 30, y:80, w: 80, h:10},
            {x: 150, y:80, w: 80, h:10}
    ],
    items: [
        {x: 100, y: 150, shape: 'triangle', color: 'green'},
        {x: 150, y: 210, shape: 'triangle', color: 'red'}
    ]
};

world.onClick = function(x, y){
    this.target.x = x;
    this.target.y = y;
    console.log("Target set:", x, y);
};

/**
 * This step function is called by the renderer every X ms.
 */
world.stepEnv = function(){
    var dx = this.target.x-this.agent.x;
    var dy = this.target.y-this.agent.y;
    dx *= this.gains; dy *= this.gains;
    if(dx*dx < 0.5) dx = 0;
    if(dy*dy < 0.5) dy = 0;

    var adjust = this.checkBounds(dx, dy);
    this.agent.x += adjust[0];
    this.agent.y += adjust[1];
};

world.checkBounds = function(dx, dy){
    var agentx = this.agent.x + dx;
    var agenty = this.agent.y + dy;

    if(agentx > this.width || agentx<0) {
        return [0, dy];
    }else if(agenty > this.height || agenty<0){
        return [dx, 0];
    }

    //Check intersections
    for(var wall in this.walls){
        wall = this.walls[wall];
        var segments = [
            {x1: wall.x, y1: wall.y, x2: wall.x+wall.w, y2:wall.y, orient:1}
            //{x1: wall.x, y1: wall.y, x2: wall.x, y2:wall.y+wall.h}
        ];

        for(var segment in segments){
            segment = segments[segment];

            var v1 = [agentx-segment.x1, agenty-segment.y1];
            var v2 = [segment.x2-segment.x1, segment.y2-segment.y1];
            var v1dv2 = v1[0]*v2[0]+v1[1]*v2[1];
            var v1norm = Math.sqrt(v1[0]*v1[0]+v1[1]*v1[1]);
            var v2norm = Math.sqrt(v2[0]*v2[0]+v2[1]*v2[1]);

            var v2hat = [v2[0]/v2norm, v2[1]/v2norm];

            var scale = v1dv2/(v2norm);
            scale = clip(scale, 0, v2norm);
            var v2proj = [v2hat[0]*scale, v2hat[1]*scale];

            var collision_pt = [segment.x1+v2proj[0], segment.y1+v2proj[1]];
            var dist = [agentx-collision_pt[0], agenty-collision_pt[1]];
            var distnorm = Math.sqrt(dist[0]*dist[0]+dist[1]*dist[1]);

            if(distnorm < this.agent.r ){
                if(segment.orient == 1) {
                    return [dx, 0];
                }else{
                    return [0, dy];
                }
            }
        }
    };
    return [dx, dy];
};


/**
 * getState is called by the renderer to update the display once during each loop
 * @returns {Array}
 */
world.getState = function(){
    var objects = [];
    objects.push({x: this.agent.x, y: this.agent.y, r:this.agent.r, type: 'agent'});
    this.walls.forEach(function(wall){
        wall.type = 'wall';
        objects.push(wall);
    });
    this.items.forEach(function(obj){
        obj.type = 'item';
        objects.push(obj);
    });
    return objects;
};

export default world;
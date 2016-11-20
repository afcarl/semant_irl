/**
 * Holds the SVG component
 * Controls the step-render loop.
 */

import React from 'react';

import Agent from './objects/agent.jsx';
import Wall from './objects/wall.jsx';
import Item from './objects/item.jsx';
import env from '../model/world.jsx';

var World = React.createClass({
    getInitialState: function(){
        return {objects:env.getState()};
    },

    componentWillMount: function(){
        var this_ = this;

        // This initializes the step-render loop.
        setInterval(function(){
            env.stepEnv();
            this_.setState({ //Render is triggered automatically by setState
                objects: env.getState()
            })
        }, this.props.stepms);
    },

    onClick: function(ev){
        env.onClick(ev.nativeEvent.offsetX,
                    ev.nativeEvent.offsetY);
    },

    render: function () {
        var obj_list = this.state.objects.map(function(obj, idx){
            if(obj.type == 'agent'){
                return <Agent data={obj} key={idx}/>
            }else if (obj.type == 'wall'){
                return <Wall data={obj} key={idx}/>
            }else if (obj.type == 'item'){
                return <Item data={obj} key={idx}/>
            }
            return null;
        });

        return <svg onClick={this.onClick} width={this.props.width} height={this.props.height}>
            {obj_list}
        </svg>;
    }
});

export default World;

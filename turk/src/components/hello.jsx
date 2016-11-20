/**
 * Main page
 */

import React from 'react';

import World from './world.jsx'

var Hello = React.createClass({
    render: function () {
        return <div>
            Execute the command:
            <b> Move to the red circle </b>
            <br />
            <World width="400" height="400" stepms="50"/>
        </div>;
    }
});

export default Hello

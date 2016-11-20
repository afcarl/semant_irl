import React from 'react';

var Item = React.createClass({
    render: function () {
        return <g transform={"translate("+this.props.data.x+","+this.props.data.y+")"}
            style={{fill: this.props.data.color}}>
            <circle cx="0" cy="0" r="8"></circle>
        </g>;
    }
});

export default Item;

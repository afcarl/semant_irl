import React from 'react';

var Agent = React.createClass({
    render: function () {
        return <rect x={this.props.data.x} y={this.props.data.y}
                width={this.props.data.w} height={this.props.data.h}>
        </rect>;
    }
});

export default Agent;

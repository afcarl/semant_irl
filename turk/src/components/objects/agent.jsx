import React from 'react';

var Agent = React.createClass({
    render: function () {
        return <circle
            className="agent"
            cx={this.props.data.x} cy={this.props.data.y} r={this.props.data.r}>
        </circle>;
    }
});

export default Agent;

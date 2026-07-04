struct Node
{
    int value;
    Node *next;

    Node(int value)
    {
        this->value = value;
    }
    Node() {}
}

Node *
reverse(Node *head)
{
    Node *curr = head;
    Node *prev = null;

    while (curr != null)
    {
        Node *temp = curr.next;
        curr->next = prev;
        prev = curr;
        curr = temp;
    }
    return prev;
}